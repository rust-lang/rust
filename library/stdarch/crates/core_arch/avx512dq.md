<summary>["AVX512DQ"]</summary><p>

[Intel's List](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#avx512techs=AVX512DQ)

- And:
    * [x] _mm_mask_and_pd
    * [x] _mm_maskz_and_pd
    * [x] _mm_mask_and_ps
    * [x] _mm_maskz_and_ps
    * [x] _mm256_mask_and_pd
    * [x] _mm256_maskz_and_pd
    * [x] _mm256_mask_and_ps
    * [x] _mm256_maskz_and_ps
    * [x] _mm512_and_pd
    * [x] _mm512_mask_and_pd
    * [x] _mm512_maskz_and_pd
    * [x] _mm512_and_ps
    * [x] _mm512_mask_and_ps
    * [x] _mm512_maskz_and_ps


- AndNot:
    * [x] _mm_mask_andnot_pd
    * [x] _mm_maskz_andnot_pd
    * [x] _mm_mask_andnot_ps
    * [x] _mm_maskz_andnot_ps
    * [x] _mm256_mask_andnot_pd
    * [x] _mm256_maskz_andnot_pd
    * [x] _mm256_mask_andnot_ps
    * [x] _mm256_maskz_andnot_ps
    * [x] _mm512_andnot_pd
    * [x] _mm512_mask_andnot_pd
    * [x] _mm512_maskz_andnot_pd
    * [x] _mm512_andnot_ps
    * [x] _mm512_mask_andnot_ps
    * [x] _mm512_maskz_andnot_ps


- Or:
    * [x] _mm_mask_or_pd
    * [x] _mm_maskz_or_pd
    * [x] _mm_mask_or_ps
    * [x] _mm_maskz_or_ps
    * [x] _mm256_mask_or_pd
    * [x] _mm256_maskz_or_pd
    * [x] _mm256_mask_or_ps
    * [x] _mm256_maskz_or_ps
    * [x] _mm512_or_pd
    * [x] _mm512_mask_or_pd
    * [x] _mm512_maskz_or_pd
    * [x] _mm512_or_ps
    * [x] _mm512_mask_or_ps
    * [x] _mm512_maskz_or_ps


- Xor:
    * [x] _mm_mask_xor_pd
    * [x] _mm_maskz_xor_pd
    * [x] _mm_mask_xor_ps
    * [x] _mm_maskz_xor_ps
    * [x] _mm256_mask_xor_pd
    * [x] _mm256_maskz_xor_pd
    * [x] _mm256_mask_xor_ps
    * [x] _mm256_maskz_xor_ps
    * [x] _mm512_xor_pd
    * [x] _mm512_mask_xor_pd
    * [x] _mm512_maskz_xor_pd
    * [x] _mm512_xor_ps
    * [x] _mm512_mask_xor_ps
    * [x] _mm512_maskz_xor_ps


- Broadcast
    * [x] _mm256_broadcast_f32x2
    * [x] _mm256_mask_broadcast_f32x2
    * [x] _mm256_maskz_broadcast_f32x2
    * [x] _mm512_broadcast_f32x2
    * [x] _mm512_mask_broadcast_f32x2
    * [x] _mm512_maskz_broadcast_f32x2
    * [x] _mm512_broadcast_f32x8
    * [x] _mm512_mask_broadcast_f32x8
    * [x] _mm512_maskz_broadcast_f32x8
    * [x] _mm256_broadcast_f64x2
    * [x] _mm256_mask_broadcast_f64x2
    * [x] _mm256_maskz_broadcast_f64x2
    * [x] _mm512_broadcast_f64x2
    * [x] _mm512_mask_broadcast_f64x2
    * [x] _mm512_maskz_broadcast_f64x2
    * [x] _mm_broadcast_i32x2
    * [x] _mm_mask_broadcast_i32x2
    * [x] _mm_maskz_broadcast_i32x2
    * [x] _mm256_broadcast_i32x2
    * [x] _mm256_mask_broadcast_i32x2
    * [x] _mm256_maskz_broadcast_i32x2
    * [x] _mm512_broadcast_i32x2
    * [x] _mm512_mask_broadcast_i32x2
    * [x] _mm512_maskz_broadcast_i32x2
    * [x] _mm512_broadcast_i32x8
    * [x] _mm512_mask_broadcast_i32x8
    * [x] _mm512_maskz_broadcast_i32x8
    * [x] _mm256_broadcast_i64x2
    * [x] _mm256_mask_broadcast_i64x2
    * [x] _mm256_maskz_broadcast_i64x2
    * [x] _mm512_broadcast_i64x2
    * [x] _mm512_mask_broadcast_i64x2
    * [x] _mm512_maskz_broadcast_i64x2


- Convert:
    * _mm512_cvt_roundepi64_pd (not in LLVM)
    * _mm512_mask_cvt_roundepi64_pd (not in LLVM)
    * _mm512_maskz_cvt_roundepi64_pd (not in LLVM)
    * _mm_cvtepi64_pd (not in LLVM)
    * _mm_mask_cvtepi64_pd (not in LLVM)
    * _mm_maskz_cvtepi64_pd (not in LLVM)
    * _mm256_cvtepi64_pd (not in LLVM)
    * _mm256_mask_cvtepi64_pd (not in LLVM)
    * _mm256_maskz_cvtepi64_pd (not in LLVM)
    * _mm512_cvtepi64_pd (not in LLVM)
    * _mm512_mask_cvtepi64_pd (not in LLVM)
    * _mm512_maskz_cvtepi64_pd (not in LLVM)
    * _mm512_cvt_roundepi64_ps (not in LLVM)
    * _mm512_mask_cvt_roundepi64_ps (not in LLVM)
    * _mm512_maskz_cvt_roundepi64_ps (not in LLVM)
    * [ ] _mm_cvtepi64_ps
    * [ ] _mm_mask_cvtepi64_ps
    * [ ] _mm_maskz_cvtepi64_ps
    * _mm256_cvtepi64_ps (not in LLVM)
    * _mm256_mask_cvtepi64_ps (not in LLVM)
    * _mm256_maskz_cvtepi64_ps (not in LLVM)
    * _mm512_cvtepi64_ps (not in LLVM)
    * _mm512_mask_cvtepi64_ps (not in LLVM)
    * _mm512_maskz_cvtepi64_ps (not in LLVM)
    * _mm512_cvt_roundepu64_pd (not in LLVM)
    * _mm512_mask_cvt_roundepu64_pd (not in LLVM)
    * _mm512_maskz_cvt_roundepu64_pd (not in LLVM)
    * _mm_cvtepu64_pd (not in LLVM)
    * _mm_mask_cvtepu64_pd (not in LLVM)
    * _mm_maskz_cvtepu64_pd (not in LLVM)
    * _mm256_cvtepu64_pd (not in LLVM)
    * _mm256_mask_cvtepu64_pd (not in LLVM)
    * _mm256_maskz_cvtepu64_pd (not in LLVM)
    * _mm512_cvtepu64_pd (not in LLVM)
    * _mm512_mask_cvtepu64_pd (not in LLVM)
    * _mm512_maskz_cvtepu64_pd (not in LLVM)
    * _mm512_cvt_roundepu64_ps (not in LLVM)
    * _mm512_mask_cvt_roundepu64_ps (not in LLVM)
    * _mm512_maskz_cvt_roundepu64_ps (not in LLVM)
    * [ ] _mm_cvtepu64_ps
    * [ ] _mm_mask_cvtepu64_ps
    * [ ] _mm_maskz_cvtepu64_ps
    * _mm256_cvtepu64_ps (not in LLVM)
    * _mm256_mask_cvtepu64_ps (not in LLVM)
    * _mm256_maskz_cvtepu64_ps (not in LLVM)
    * _mm512_cvtepu64_ps (not in LLVM)
    * _mm512_mask_cvtepu64_ps (not in LLVM)
    * _mm512_maskz_cvtepu64_ps (not in LLVM)
    * [ ] _mm512_cvt_roundpd_epi64
    * [ ] _mm512_mask_cvt_roundpd_epi64
    * [ ] _mm512_maskz_cvt_roundpd_epi64
    * [ ] _mm_cvtpd_epi64
    * [ ] _mm_mask_cvtpd_epi64
    * [ ] _mm_maskz_cvtpd_epi64
    * [ ] _mm256_cvtpd_epi64
    * [ ] _mm256_mask_cvtpd_epi64
    * [ ] _mm256_maskz_cvtpd_epi64
    * [ ] _mm512_cvtpd_epi64
    * [ ] _mm512_mask_cvtpd_epi64
    * [ ] _mm512_maskz_cvtpd_epi64
    * [ ] _mm512_cvt_roundpd_epu64
    * [ ] _mm512_mask_cvt_roundpd_epu64
    * [ ] _mm512_maskz_cvt_roundpd_epu64
    * [ ] _mm_cvtpd_epu64
    * [ ] _mm_mask_cvtpd_epu64
    * [ ] _mm_maskz_cvtpd_epu64
    * [ ] _mm256_cvtpd_epu64
    * [ ] _mm256_mask_cvtpd_epu64
    * [ ] _mm256_maskz_cvtpd_epu64
    * [ ] _mm512_cvtpd_epu64
    * [ ] _mm512_mask_cvtpd_epu64
    * [ ] _mm512_maskz_cvtpd_epu64
    * [ ] _mm512_cvt_roundps_epi64
    * [ ] _mm512_mask_cvt_roundps_epi64
    * [ ] _mm512_maskz_cvt_roundps_epi64
    * [ ] _mm_cvtps_epi64
    * [ ] _mm_mask_cvtps_epi64
    * [ ] _mm_maskz_cvtps_epi64
    * [ ] _mm256_cvtps_epi64
    * [ ] _mm256_mask_cvtps_epi64
    * [ ] _mm256_maskz_cvtps_epi64
    * [ ] _mm512_cvtps_epi64
    * [ ] _mm512_mask_cvtps_epi64
    * [ ] _mm512_maskz_cvtps_epi64
    * [ ] _mm512_cvt_roundps_epu64
    * [ ] _mm512_mask_cvt_roundps_epu64
    * [ ] _mm512_maskz_cvt_roundps_epu64
    * [ ] _mm_cvtps_epu64
    * [ ] _mm_mask_cvtps_epu64
    * [ ] _mm_maskz_cvtps_epu64
    * [ ] _mm256_cvtps_epu64
    * [ ] _mm256_mask_cvtps_epu64
    * [ ] _mm256_maskz_cvtps_epu64
    * [ ] _mm512_cvtps_epu64
    * [ ] _mm512_mask_cvtps_epu64
    * [ ] _mm512_maskz_cvtps_epu64
    * [ ] _mm512_cvtt_roundpd_epi64
    * [ ] _mm512_mask_cvtt_roundpd_epi64
    * [ ] _mm512_maskz_cvtt_roundpd_epi64
    * [ ] _mm_cvttpd_epi64
    * [ ] _mm_mask_cvttpd_epi64
    * [ ] _mm_maskz_cvttpd_epi64
    * [ ] _mm256_cvttpd_epi64
    * [ ] _mm256_mask_cvttpd_epi64
    * [ ] _mm256_maskz_cvttpd_epi64
    * [ ] _mm512_cvttpd_epi64
    * [ ] _mm512_mask_cvttpd_epi64
    * [ ] _mm512_maskz_cvttpd_epi64
    * [ ] _mm512_cvtt_roundpd_epu64
    * [ ] _mm512_mask_cvtt_roundpd_epu64
    * [ ] _mm512_maskz_cvtt_roundpd_epu64
    * [ ] _mm_cvttpd_epu64
    * [ ] _mm_mask_cvttpd_epu64
    * [ ] _mm_maskz_cvttpd_epu64
    * [ ] _mm256_cvttpd_epu64
    * [ ] _mm256_mask_cvttpd_epu64
    * [ ] _mm256_maskz_cvttpd_epu64
    * [ ] _mm512_cvttpd_epu64
    * [ ] _mm512_mask_cvttpd_epu64
    * [ ] _mm512_maskz_cvttpd_epu64
    * [ ] _mm512_cvtt_roundps_epi64
    * [ ] _mm512_mask_cvtt_roundps_epi64
    * [ ] _mm512_maskz_cvtt_roundps_epi64
    * [ ] _mm_cvttps_epi64
    * [ ] _mm_mask_cvttps_epi64
    * [ ] _mm_maskz_cvttps_epi64
    * [ ] _mm256_cvttps_epi64
    * [ ] _mm256_mask_cvttps_epi64
    * [ ] _mm256_maskz_cvttps_epi64
    * [ ] _mm512_cvttps_epi64
    * [ ] _mm512_mask_cvttps_epi64
    * [ ] _mm512_maskz_cvttps_epi64
    * [ ] _mm512_cvtt_roundps_epu64
    * [ ] _mm512_mask_cvtt_roundps_epu64
    * [ ] _mm512_maskz_cvtt_roundps_epu64
    * [ ] _mm_cvttps_epu64
    * [ ] _mm_mask_cvttps_epu64
    * [ ] _mm_maskz_cvttps_epu64
    * [ ] _mm256_cvttps_epu64
    * [ ] _mm256_mask_cvttps_epu64
    * [ ] _mm256_maskz_cvttps_epu64
    * [ ] _mm512_cvttps_epu64
    * [ ] _mm512_mask_cvttps_epu64
    * [ ] _mm512_maskz_cvttps_epu64


- Element Extract:
    * [x] _mm512_extractf32x8_ps
    * [x] _mm512_mask_extractf32x8_ps
    * [x] _mm512_maskz_extractf32x8_ps
    * [x] _mm256_extractf64x2_pd
    * [x] _mm256_mask_extractf64x2_pd
    * [x] _mm256_maskz_extractf64x2_pd
    * [x] _mm512_extractf64x2_pd
    * [x] _mm512_mask_extractf64x2_pd
    * [x] _mm512_maskz_extractf64x2_pd
    * [x] _mm512_extracti32x8_epi32
    * [x] _mm512_mask_extracti32x8_epi32
    * [x] _mm512_maskz_extracti32x8_epi32
    * [x] _mm256_extracti64x2_epi64
    * [x] _mm256_mask_extracti64x2_epi64
    * [x] _mm256_maskz_extracti64x2_epi64
    * [x] _mm512_extracti64x2_epi64
    * [x] _mm512_mask_extracti64x2_epi64
    * [x] _mm512_maskz_extracti64x2_epi64


- Element Insert:
    * [x] _mm512_insertf32x8
    * [x] _mm512_mask_insertf32x8
    * [x] _mm512_maskz_insertf32x8
    * [x] _mm256_insertf64x2
    * [x] _mm256_mask_insertf64x2
    * [x] _mm256_maskz_insertf64x2
    * [x] _mm512_insertf64x2
    * [x] _mm512_mask_insertf64x2
    * [x] _mm512_maskz_insertf64x2
    * [x] _mm512_inserti32x8
    * [x] _mm512_mask_inserti32x8
    * [x] _mm512_maskz_inserti32x8
    * [x] _mm256_inserti64x2
    * [x] _mm256_mask_inserti64x2
    * [x] _mm256_maskz_inserti64x2
    * [x] _mm512_inserti64x2
    * [x] _mm512_mask_inserti64x2
    * [x] _mm512_maskz_inserti64x2


- FP-Class
    * [ ] _mm_fpclass_pd_mask
    * [ ] _mm_mask_fpclass_pd_mask
    * [ ] _mm256_fpclass_pd_mask
    * [ ] _mm256_mask_fpclass_pd_mask
    * [ ] _mm512_fpclass_pd_mask
    * [ ] _mm512_mask_fpclass_pd_mask
    * [ ] _mm_fpclass_ps_mask
    * [ ] _mm_mask_fpclass_ps_mask
    * [ ] _mm256_fpclass_ps_mask
    * [ ] _mm256_mask_fpclass_ps_mask
    * [ ] _mm512_fpclass_ps_mask
    * [ ] _mm512_mask_fpclass_ps_mask
    * [ ] _mm_fpclass_sd_mask
    * [ ] _mm_mask_fpclass_sd_mask
    * [ ] _mm_fpclass_ss_mask
    * [ ] _mm_mask_fpclass_ss_mask


- Mask Registers
    * [ ] _cvtmask8_u32
    * [ ] _cvtu32_mask8
    * [ ] _kadd_mask16
    * [ ] _kadd_mask8
    * [ ] _kand_mask8
    * [ ] _kandn_mask8
    * [ ] _knot_mask8
    * [ ] _kor_mask8
    * [ ] _kortest_mask8_u8
    * [ ] _kortestc_mask8_u8
    * [ ] _kortestz_mask8_u8
    * [ ] _kshiftli_mask8
    * [ ] _kshiftri_mask8
    * [ ] _ktest_mask16_u8
    * [ ] _ktest_mask8_u8
    * [ ] _ktestc_mask16_u8
    * [ ] _ktestc_mask8_u8
    * [ ] _ktestz_mask16_u8
    * [ ] _ktestz_mask8_u8
    * [ ] _kxnor_mask8
    * [ ] _kxor_mask8
    * [ ] _load_mask8


- Mask register for Bit patterns
    * [ ] _mm_movepi32_mask
    * [ ] _mm256_movepi32_mask
    * [ ] _mm512_movepi32_mask
    * [ ] _mm_movepi64_mask
    * [ ] _mm256_movepi64_mask
    * [ ] _mm512_movepi64_mask
    * [ ] _mm_movm_epi32
    * [ ] _mm256_movm_epi32
    * [ ] _mm512_movm_epi32
    * [ ] _mm_movm_epi64
    * [ ] _mm256_movm_epi64
    * [ ] _mm512_movm_epi64


- Multiply Low
    * _mm_mullo_epi64 (not in LLVM)
    * _mm_mask_mullo_epi64 (not in LLVM)
    * _mm_maskz_mullo_epi64 (not in LLVM)
    * _mm256_mullo_epi64 (not in LLVM)
    * _mm256_mask_mullo_epi64 (not in LLVM)
    * _mm256_maskz_mullo_epi64 (not in LLVM)
    * _mm512_mullo_epi64 (not in LLVM)
    * _mm512_mask_mullo_epi64 (not in LLVM)
    * _mm512_maskz_mullo_epi64 (not in LLVM)


- Range
    * [ ] _mm512_range_round_pd
    * [ ] _mm512_mask_range_round_pd
    * [ ] _mm512_maskz_range_round_pd
    * [ ] _mm_range_pd
    * [ ] _mm_mask_range_pd
    * [ ] _mm_maskz_range_pd
    * [ ] _mm256_range_pd
    * [ ] _mm256_mask_range_pd
    * [ ] _mm256_maskz_range_pd
    * [ ] _mm512_range_pd
    * [ ] _mm512_mask_range_pd
    * [ ] _mm512_maskz_range_pd
    * [ ] _mm512_range_round_ps
    * [ ] _mm512_mask_range_round_ps
    * [ ] _mm512_maskz_range_round_ps
    * [ ] _mm_range_ps
    * [ ] _mm_mask_range_ps
    * [ ] _mm_maskz_range_ps
    * [ ] _mm256_range_ps
    * [ ] _mm256_mask_range_ps
    * [ ] _mm256_maskz_range_ps
    * [ ] _mm512_range_ps
    * [ ] _mm512_mask_range_ps
    * [ ] _mm512_maskz_range_ps
    * [ ] _mm_range_round_sd
    * [ ] _mm_mask_range_round_sd
    * [ ] _mm_maskz_range_round_sd
    * [ ] _mm_mask_range_sd
    * [ ] _mm_maskz_range_sd
    * [ ] _mm_range_round_ss
    * [ ] _mm_mask_range_round_ss
    * [ ] _mm_maskz_range_round_ss
    * [ ] _mm_mask_range_ss
    * [ ] _mm_maskz_range_ss


- Range
    * [ ] _mm512_reduce_round_pd
    * [ ] _mm512_mask_reduce_round_pd
    * [ ] _mm512_maskz_reduce_round_pd
    * [ ] _mm_reduce_pd
    * [ ] _mm_mask_reduce_pd
    * [ ] _mm_maskz_reduce_pd
    * [ ] _mm256_reduce_pd
    * [ ] _mm256_mask_reduce_pd
    * [ ] _mm256_maskz_reduce_pd
    * [ ] _mm512_reduce_pd
    * [ ] _mm512_mask_reduce_pd
    * [ ] _mm512_maskz_reduce_pd
    * [ ] _mm512_reduce_round_ps
    * [ ] _mm512_mask_reduce_round_ps
    * [ ] _mm512_maskz_reduce_round_ps
    * [ ] _mm_reduce_ps
    * [ ] _mm_mask_reduce_ps
    * [ ] _mm_maskz_reduce_ps
    * [ ] _mm256_reduce_ps
    * [ ] _mm256_mask_reduce_ps
    * [ ] _mm256_maskz_reduce_ps
    * [ ] _mm512_reduce_ps
    * [ ] _mm512_mask_reduce_ps
    * [ ] _mm512_maskz_reduce_ps
    * [ ] _mm_reduce_round_sd
    * [ ] _mm_mask_reduce_round_sd
    * [ ] _mm_maskz_reduce_round_sd
    * [ ] _mm_reduce_sd
    * [ ] _mm_mask_reduce_sd
    * [ ] _mm_maskz_reduce_sd
    * [ ] _mm_reduce_round_ss
    * [ ] _mm_mask_reduce_round_ss
    * [ ] _mm_maskz_reduce_round_ss
    * [ ] _mm_reduce_ss
    * [ ] _mm_mask_reduce_ss
    * [ ] _mm_maskz_reduce_ss
</p>
