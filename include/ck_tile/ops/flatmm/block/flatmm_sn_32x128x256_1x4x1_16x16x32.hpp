// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm.hpp"
#include "ck_tile/ops/flatmm/block/flatmm_uk_config.hpp"

namespace ck_tile {

// "S"tream update output along "N"
// A in smem, B load from global
// require 4 wave, occupancy=1c
struct FlatmmSn_32x128x256_1x4x1_16x16x32_Base
{
    static constexpr index_t Block_M = 32;
    static constexpr index_t Block_N = 128;
    static constexpr index_t Block_K = 256;

    static constexpr index_t WarpPerBlock_M = 1;
    static constexpr index_t WarpPerBlock_N = 4;    
    static constexpr index_t WarpPerBlock_K = 1;

    static constexpr index_t Warp_M = 16;
    static constexpr index_t Warp_N = 16;
    static constexpr index_t Warp_K = 32;

    static constexpr index_t BlockSize = 256;

    // static constexpr index_t KPack = 2; // this is used to gurantee every threads can do dwordx4

    // TODO: note Nr/Kr/W need consider KPack
    static constexpr index_t Block_W  = Warp_N * Warp_K;  // 512 element
    static constexpr index_t Block_Nr = Block_N / Warp_N; // 32 element, 4 per wave
    static constexpr index_t Block_Kr = Block_K / Warp_K; // 8

    static constexpr index_t Repeat_M = Block_M / (Warp_M * WarpPerBlock_M); // 2
    static constexpr index_t Repeat_N = Block_N / (Warp_N * WarpPerBlock_N); // 2
    static constexpr index_t Repeat_K = Block_K / (Warp_K * WarpPerBlock_K); // 8

    static CK_TILE_DEVICE constexpr auto MakeCBlockDist()
    {
        constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<Repeat_M, WarpPerBlock_M>, sequence<Repeat_N, WarpPerBlock_N>>,
            tuple<sequence<1, 2>>,
            tuple<sequence<1, 1>>,
            sequence<2, 1>, // !! note here is different
            sequence<0, 0>>{};

        using WG = WarpGemmMfmaF16F16F32M16N16K32TransposedCDistribution;

        constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WG::CWarpDstrEncoding{});
        constexpr auto c_block_dstr = make_static_tile_distribution(c_block_dstr_encode);
        return c_block_dstr;
    }

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        //                    y     y     p     p      p      y
        // reg before shfl  M0(2)*N0(2)*Nl(4)*Nw(4)*Mw(16)*Nv(4)
        // but order is N0*M0*Nv
        // in LDS we need store as
        //          M0(2)* N0(2) *  Nl(4) * Nw(4) * (Mw(16)*Nv(4) + 4)
        //             y    y       wave-id  lid/16  lid%16   v
        return 2 * 2 * 4 * 4 * (16 * 4 + 4) * sizeof(bf16_t);
    }
};

} // namespace ck_tile
