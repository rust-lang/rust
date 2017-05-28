**TIP**: Use the following command to generate a section in this list for
Intel intrinsics. Replace `SSE4.2` with the intended type.

```
rg '^<intrinsic' intel-intrinsics-3.3.15.xml | rg "'SSE4.2'" | rg '^.*name=\x27([^\x27]+)\x27.*$' -r '* [ ] `$1`' >> TODO.md
```

rg calls the ripgrep tool, which can be installed with `cargo install ripgrep`

sse
---
* [ ] `_MM_TRANSPOSE4_PS`
* [ ] `_mm_getcsr`
* [ ] `_mm_setcsr`
* [ ] `_MM_GET_EXCEPTION_STATE`
* [ ] `_MM_SET_EXCEPTION_STATE`
* [ ] `_MM_GET_EXCEPTION_MASK`
* [ ] `_MM_SET_EXCEPTION_MASK`
* [ ] `_MM_GET_ROUNDING_MODE`
* [ ] `_MM_SET_ROUNDING_MODE`
* [ ] `_MM_GET_FLUSH_ZERO_MODE`
* [ ] `_MM_SET_FLUSH_ZERO_MODE`
* [ ] `_mm_prefetch`
* [ ] `_mm_sfence`
* [ ] `_mm_max_pi16`
* [ ] `_m_pmaxsw`
* [ ] `_mm_max_pu8`
* [ ] `_m_pmaxub`
* [ ] `_mm_min_pi16`
* [ ] `_m_pminsw`
* [ ] `_mm_min_pu8`
* [ ] `_m_pminub`
* [ ] `_mm_mulhi_pu16`
* [ ] `_m_pmulhuw`
* [ ] `_mm_avg_pu8`
* [ ] `_m_pavgb`
* [ ] `_mm_avg_pu16`
* [ ] `_m_pavgw`
* [ ] `_mm_sad_pu8`
* [ ] `_m_psadbw`
* [ ] `_mm_cvtsi32_ss`
* [ ] `_mm_cvt_si2ss`
* [ ] `_mm_cvtsi64_ss`
* [ ] `_mm_cvtpi32_ps`
* [ ] `_mm_cvt_pi2ps`
* [ ] `_mm_cvtpi16_ps`
* [ ] `_mm_cvtpu16_ps`
* [ ] `_mm_cvtpi8_ps`
* [ ] `_mm_cvtpu8_ps`
* [ ] `_mm_cvtpi32x2_ps`
* [ ] `_mm_stream_pi`
* [ ] `_mm_maskmove_si64`
* [ ] `_m_maskmovq`
* [ ] `_mm_extract_pi16`
* [ ] `_m_pextrw`
* [ ] `_mm_insert_pi16`
* [ ] `_m_pinsrw`
* [ ] `_mm_movemask_pi8`
* [ ] `_m_pmovmskb`
* [ ] `_mm_shuffle_pi16`
* [ ] `_m_pshufw`
* [ ] `_mm_add_ss`
* [ ] `_mm_add_ps`
* [ ] `_mm_sub_ss`
* [ ] `_mm_sub_ps`
* [ ] `_mm_mul_ss`
* [ ] `_mm_mul_ps`
* [ ] `_mm_div_ss`
* [ ] `_mm_div_ps`
* [ ] `_mm_sqrt_ss`
* [x] `_mm_sqrt_ps`
* [ ] `_mm_rcp_ss`
* [x] `_mm_rcp_ps`
* [ ] `_mm_rsqrt_ss`
* [x] `_mm_rsqrt_ps`
* [ ] `_mm_min_ss`
* [x] `_mm_min_ps`
* [ ] `_mm_max_ss`
* [x] `_mm_max_ps`
* [ ] `_mm_and_ps`
* [ ] `_mm_andnot_ps`
* [ ] `_mm_or_ps`
* [ ] `_mm_xor_ps`
* [ ] `_mm_cmpeq_ss`
* [ ] `_mm_cmpeq_ps`
* [ ] `_mm_cmplt_ss`
* [ ] `_mm_cmplt_ps`
* [ ] `_mm_cmple_ss`
* [ ] `_mm_cmple_ps`
* [ ] `_mm_cmpgt_ss`
* [ ] `_mm_cmpgt_ps`
* [ ] `_mm_cmpge_ss`
* [ ] `_mm_cmpge_ps`
* [ ] `_mm_cmpneq_ss`
* [ ] `_mm_cmpneq_ps`
* [ ] `_mm_cmpnlt_ss`
* [ ] `_mm_cmpnlt_ps`
* [ ] `_mm_cmpnle_ss`
* [ ] `_mm_cmpnle_ps`
* [ ] `_mm_cmpngt_ss`
* [ ] `_mm_cmpngt_ps`
* [ ] `_mm_cmpnge_ss`
* [ ] `_mm_cmpnge_ps`
* [ ] `_mm_cmpord_ss`
* [ ] `_mm_cmpord_ps`
* [ ] `_mm_cmpunord_ss`
* [ ] `_mm_cmpunord_ps`
* [ ] `_mm_comieq_ss`
* [ ] `_mm_comilt_ss`
* [ ] `_mm_comile_ss`
* [ ] `_mm_comigt_ss`
* [ ] `_mm_comige_ss`
* [ ] `_mm_comineq_ss`
* [ ] `_mm_ucomieq_ss`
* [ ] `_mm_ucomilt_ss`
* [ ] `_mm_ucomile_ss`
* [ ] `_mm_ucomigt_ss`
* [ ] `_mm_ucomige_ss`
* [ ] `_mm_ucomineq_ss`
* [ ] `_mm_cvtss_si32`
* [ ] `_mm_cvt_ss2si`
* [ ] `_mm_cvtss_si64`
* [ ] `_mm_cvtss_f32`
* [ ] `_mm_cvtps_pi32`
* [ ] `_mm_cvt_ps2pi`
* [ ] `_mm_cvttss_si32`
* [ ] `_mm_cvtt_ss2si`
* [ ] `_mm_cvttss_si64`
* [ ] `_mm_cvttps_pi32`
* [ ] `_mm_cvtt_ps2pi`
* [ ] `_mm_cvtps_pi16`
* [ ] `_mm_cvtps_pi8`
* [ ] `_mm_set_ss`
* [ ] `_mm_set1_ps`
* [ ] `_mm_set_ps1`
* [ ] `_mm_set_ps`
* [ ] `_mm_setr_ps`
* [ ] `_mm_setzero_ps`
* [ ] `_mm_loadh_pi`
* [ ] `_mm_loadl_pi`
* [ ] `_mm_load_ss`
* [ ] `_mm_load1_ps`
* [ ] `_mm_load_ps1`
* [ ] `_mm_load_ps`
* [ ] `_mm_loadu_ps`
* [ ] `_mm_loadr_ps`
* [ ] `_mm_stream_ps`
* [ ] `_mm_storeh_pi`
* [ ] `_mm_storel_pi`
* [ ] `_mm_store_ss`
* [ ] `_mm_store1_ps`
* [ ] `_mm_store_ps1`
* [ ] `_mm_store_ps`
* [ ] `_mm_storeu_ps`
* [ ] `_mm_storer_ps`
* [ ] `_mm_move_ss`
* [ ] `_mm_shuffle_ps`
* [ ] `_mm_unpackhi_ps`
* [ ] `_mm_unpacklo_ps`
* [ ] `_mm_movehl_ps`
* [ ] `_mm_movelh_ps`
* [x] `_mm_movemask_ps`
* [ ] `_mm_undefined_ps`


sse2
----
* [x] `_mm_pause`
* [x] `_mm_clflush`
* [x] `_mm_lfence`
* [x] `_mm_mfence`
* [x] `_mm_add_epi8`
* [x] `_mm_add_epi16`
* [x] `_mm_add_epi32`
* [ ] `_mm_add_si64`
* [x] `_mm_add_epi64`
* [x] `_mm_adds_epi8`
* [x] `_mm_adds_epi16`
* [x] `_mm_adds_epu8`
* [x] `_mm_adds_epu16`
* [x] `_mm_avg_epu8`
* [x] `_mm_avg_epu16`
* [x] `_mm_madd_epi16`
* [x] `_mm_max_epi16`
* [x] `_mm_max_epu8`
* [x] `_mm_min_epi16`
* [x] `_mm_min_epu8`
* [x] `_mm_mulhi_epi16`
* [x] `_mm_mulhi_epu16`
* [x] `_mm_mullo_epi16`
* [ ] `_mm_mul_su32`
* [x] `_mm_mul_epu32`
* [x] `_mm_sad_epu8`
* [x] `_mm_sub_epi8`
* [x] `_mm_sub_epi16`
* [x] `_mm_sub_epi32`
* [ ] `_mm_sub_si64`
* [x] `_mm_sub_epi64`
* [x] `_mm_subs_epi8`
* [x] `_mm_subs_epi16`
* [x] `_mm_subs_epu8`
* [x] `_mm_subs_epu16`
* [x] `_mm_slli_si128`
* [x] `_mm_bslli_si128`
* [x] `_mm_bsrli_si128`
* [x] `_mm_slli_epi16`
* [x] `_mm_sll_epi16`
* [x] `_mm_slli_epi32`
* [x] `_mm_sll_epi32`
* [x] `_mm_slli_epi64`
* [x] `_mm_sll_epi64`
* [x] `_mm_srai_epi16`
* [x] `_mm_sra_epi16`
* [x] `_mm_srai_epi32`
* [x] `_mm_sra_epi32`
* [x] `_mm_srli_si128`
* [x] `_mm_srli_epi16`
* [x] `_mm_srl_epi16`
* [x] `_mm_srli_epi32`
* [x] `_mm_srl_epi32`
* [x] `_mm_srli_epi64`
* [x] `_mm_srl_epi64`
* [x] `_mm_and_si128`
* [x] `_mm_andnot_si128`
* [x] `_mm_or_si128`
* [x] `_mm_xor_si128`
* [x] `_mm_cmpeq_epi8`
* [x] `_mm_cmpeq_epi16`
* [x] `_mm_cmpeq_epi32`
* [x] `_mm_cmpgt_epi8`
* [x] `_mm_cmpgt_epi16`
* [x] `_mm_cmpgt_epi32`
* [x] `_mm_cmplt_epi8`
* [x] `_mm_cmplt_epi16`
* [x] `_mm_cmplt_epi32`
* [x] `_mm_cvtepi32_pd`
* [x] `_mm_cvtsi32_sd`
* [x] `_mm_cvtsi64_sd`
* [x] `_mm_cvtsi64x_sd`
* [x] `_mm_cvtepi32_ps`
* [ ] `_mm_cvtpi32_pd`
* [x] `_mm_cvtsi32_si128`
* [x] `_mm_cvtsi64_si128`
* [x] `_mm_cvtsi64x_si128`
* [x] `_mm_cvtsi128_si32`
* [x] `_mm_cvtsi128_si64`
* [x] `_mm_cvtsi128_si64x`
* [ ] `_mm_set_epi64`
* [x] `_mm_set_epi64x`
* [x] `_mm_set_epi32`
* [x] `_mm_set_epi16`
* [x] `_mm_set_epi8`
* [ ] `_mm_set1_epi64`
* [x] `_mm_set1_epi64x`
* [x] `_mm_set1_epi32`
* [x] `_mm_set1_epi16`
* [x] `_mm_set1_epi8`
* [ ] `_mm_setr_epi64`
* [x] `_mm_setr_epi32`
* [x] `_mm_setr_epi16`
* [x] `_mm_setr_epi8`
* [x] `_mm_setzero_si128`
* [x] `_mm_loadl_epi64`
* [x] `_mm_load_si128`
* [x] `_mm_loadu_si128`
* [x] `_mm_maskmoveu_si128`
* [x] `_mm_store_si128`
* [x] `_mm_storeu_si128`
* [x] `_mm_storel_epi64`
* [ ] `_mm_stream_si128`
* [ ] `_mm_stream_si32`
* [ ] `_mm_stream_si64`
* [ ] `_mm_movepi64_pi64`
* [ ] `_mm_movpi64_epi64`
* [x] `_mm_move_epi64`
* [x] `_mm_packs_epi16`
* [x] `_mm_packs_epi32`
* [x] `_mm_packus_epi16`
* [x] `_mm_extract_epi16`
* [x] `_mm_insert_epi16`
* [x] `_mm_movemask_epi8`
* [x] `_mm_shuffle_epi32`
* [x] `_mm_shufflehi_epi16`
* [x] `_mm_shufflelo_epi16`
* [x] `_mm_unpackhi_epi8`
* [x] `_mm_unpackhi_epi16`
* [x] `_mm_unpackhi_epi32`
* [x] `_mm_unpackhi_epi64`
* [x] `_mm_unpacklo_epi8`
* [x] `_mm_unpacklo_epi16`
* [x] `_mm_unpacklo_epi32`
* [x] `_mm_unpacklo_epi64`
* [x] `_mm_add_sd`
* [x] `_mm_add_pd`
* [x] `_mm_div_sd`
* [x] `_mm_div_pd`
* [x] `_mm_max_sd`
* [x] `_mm_max_pd`
* [x] `_mm_min_sd`
* [x] `_mm_min_pd`
* [x] `_mm_mul_sd`
* [x] `_mm_mul_pd`
* [x] `_mm_sqrt_sd`
* [x] `_mm_sqrt_pd`
* [x] `_mm_sub_sd`
* [x] `_mm_sub_pd`
* [x] `_mm_and_pd`
* [x] `_mm_andnot_pd`
* [x] `_mm_or_pd`
* [x] `_mm_xor_pd`
* [x] `_mm_cmpeq_sd`
* [x] `_mm_cmplt_sd`
* [x] `_mm_cmple_sd`
* [x] `_mm_cmpgt_sd`
* [x] `_mm_cmpge_sd`
* [x] `_mm_cmpord_sd`
* [x] `_mm_cmpunord_sd`
* [x] `_mm_cmpneq_sd`
* [x] `_mm_cmpnlt_sd`
* [x] `_mm_cmpnle_sd`
* [x] `_mm_cmpngt_sd`
* [x] `_mm_cmpnge_sd`
* [x] `_mm_cmpeq_pd`
* [x] `_mm_cmplt_pd`
* [x] `_mm_cmple_pd`
* [x] `_mm_cmpgt_pd`
* [x] `_mm_cmpge_pd`
* [x] `_mm_cmpord_pd`
* [x] `_mm_cmpunord_pd`
* [x] `_mm_cmpneq_pd`
* [x] `_mm_cmpnlt_pd`
* [x] `_mm_cmpnle_pd`
* [x] `_mm_cmpngt_pd`
* [x] `_mm_cmpnge_pd`
* [x] `_mm_comieq_sd`
* [x] `_mm_comilt_sd`
* [x] `_mm_comile_sd`
* [x] `_mm_comigt_sd`
* [x] `_mm_comige_sd`
* [x] `_mm_comineq_sd`
* [x] `_mm_ucomieq_sd`
* [x] `_mm_ucomilt_sd`
* [x] `_mm_ucomile_sd`
* [x] `_mm_ucomigt_sd`
* [x] `_mm_ucomige_sd`
* [x] `_mm_ucomineq_sd`
* [ ] `_mm_cvtpd_ps`
* [ ] `_mm_cvtps_pd`
* [ ] `_mm_cvtpd_epi32`
* [ ] `_mm_cvtsd_si32`
* [ ] `_mm_cvtsd_si64`
* [ ] `_mm_cvtsd_si64x`
* [ ] `_mm_cvtsd_ss`
* [ ] `_mm_cvtsd_f64`
* [ ] `_mm_cvtss_sd`
* [ ] `_mm_cvttpd_epi32`
* [ ] `_mm_cvttsd_si32`
* [ ] `_mm_cvttsd_si64`
* [ ] `_mm_cvttsd_si64x`
* [ ] `_mm_cvtps_epi32`
* [ ] `_mm_cvttps_epi32`
* [ ] `_mm_cvtpd_pi32`
* [ ] `_mm_cvttpd_pi32`
* [ ] `_mm_set_sd`
* [ ] `_mm_set1_pd`
* [ ] `_mm_set_pd1`
* [ ] `_mm_set_pd`
* [ ] `_mm_setr_pd`
* [ ] `_mm_setzero_pd`
* [ ] `_mm_load_pd`
* [ ] `_mm_load1_pd`
* [ ] `_mm_load_pd1`
* [ ] `_mm_loadr_pd`
* [ ] `_mm_loadu_pd`
* [ ] `_mm_load_sd`
* [ ] `_mm_loadh_pd`
* [ ] `_mm_loadl_pd`
* [ ] `_mm_stream_pd`
* [ ] `_mm_store_sd`
* [ ] `_mm_store1_pd`
* [ ] `_mm_store_pd1`
* [ ] `_mm_store_pd`
* [ ] `_mm_storeu_pd`
* [ ] `_mm_storer_pd`
* [ ] `_mm_storeh_pd`
* [ ] `_mm_storel_pd`
* [ ] `_mm_unpackhi_pd`
* [ ] `_mm_unpacklo_pd`
* [x] `_mm_movemask_pd`
* [ ] `_mm_shuffle_pd`
* [ ] `_mm_move_sd`
* [ ] `_mm_castpd_ps`
* [ ] `_mm_castpd_si128`
* [ ] `_mm_castps_pd`
* [ ] `_mm_castps_si128`
* [ ] `_mm_castsi128_pd`
* [ ] `_mm_castsi128_ps`
* [ ] `_mm_undefined_pd`
* [ ] `_mm_undefined_si128`


sse3
----
* [ ] `_mm_addsub_ps`
* [ ] `_mm_addsub_pd`
* [ ] `_mm_hadd_pd`
* [ ] `_mm_hadd_ps`
* [ ] `_mm_hsub_pd`
* [ ] `_mm_hsub_ps`
* [ ] `_mm_lddqu_si128`
* [ ] `_mm_movedup_pd`
* [ ] `_mm_loaddup_pd`
* [ ] `_mm_movehdup_ps`
* [ ] `_mm_moveldup_ps`


ssse3
-----
* [ ] `_mm_abs_pi8`
* [x] `_mm_abs_epi8`
* [ ] `_mm_abs_pi16`
* [ ] `_mm_abs_epi16`
* [ ] `_mm_abs_pi32`
* [ ] `_mm_abs_epi32`
* [ ] `_mm_shuffle_epi8`
* [ ] `_mm_shuffle_pi8`
* [ ] `_mm_alignr_epi8`
* [ ] `_mm_alignr_pi8`
* [ ] `_mm_hadd_epi16`
* [ ] `_mm_hadds_epi16`
* [ ] `_mm_hadd_epi32`
* [ ] `_mm_hadd_pi16`
* [ ] `_mm_hadd_pi32`
* [ ] `_mm_hadds_pi16`
* [ ] `_mm_hsub_epi16`
* [ ] `_mm_hsubs_epi16`
* [ ] `_mm_hsub_epi32`
* [ ] `_mm_hsub_pi16`
* [ ] `_mm_hsub_pi32`
* [ ] `_mm_hsubs_pi16`
* [ ] `_mm_maddubs_epi16`
* [ ] `_mm_maddubs_pi16`
* [ ] `_mm_mulhrs_epi16`
* [ ] `_mm_mulhrs_pi16`
* [ ] `_mm_sign_epi8`
* [ ] `_mm_sign_epi16`
* [ ] `_mm_sign_epi32`
* [ ] `_mm_sign_pi8`
* [ ] `_mm_sign_pi16`
* [ ] `_mm_sign_pi32`


sse4.1
------
* [ ] `_mm_blend_pd`
* [ ] `_mm_blend_ps`
* [ ] `_mm_blendv_pd`
* [ ] `_mm_blendv_ps`
* [ ] `_mm_blendv_epi8`
* [ ] `_mm_blend_epi16`
* [ ] `_mm_dp_pd`
* [ ] `_mm_dp_ps`
* [ ] `_mm_extract_ps`
* [ ] `_mm_extract_epi8`
* [ ] `_mm_extract_epi32`
* [ ] `_mm_extract_epi64`
* [ ] `_mm_insert_ps`
* [ ] `_mm_insert_epi8`
* [ ] `_mm_insert_epi32`
* [ ] `_mm_insert_epi64`
* [ ] `_mm_max_epi8`
* [ ] `_mm_max_epi32`
* [ ] `_mm_max_epu32`
* [ ] `_mm_max_epu16`
* [ ] `_mm_min_epi8`
* [ ] `_mm_min_epi32`
* [ ] `_mm_min_epu32`
* [ ] `_mm_min_epu16`
* [ ] `_mm_packus_epi32`
* [ ] `_mm_cmpeq_epi64`
* [ ] `_mm_cvtepi8_epi16`
* [ ] `_mm_cvtepi8_epi32`
* [ ] `_mm_cvtepi8_epi64`
* [ ] `_mm_cvtepi16_epi32`
* [ ] `_mm_cvtepi16_epi64`
* [ ] `_mm_cvtepi32_epi64`
* [ ] `_mm_cvtepu8_epi16`
* [ ] `_mm_cvtepu8_epi32`
* [ ] `_mm_cvtepu8_epi64`
* [ ] `_mm_cvtepu16_epi32`
* [ ] `_mm_cvtepu16_epi64`
* [ ] `_mm_cvtepu32_epi64`
* [ ] `_mm_mul_epi32`
* [ ] `_mm_mullo_epi32`
* [ ] `_mm_testz_si128`
* [ ] `_mm_testc_si128`
* [ ] `_mm_testnzc_si128`
* [ ] `_mm_test_all_zeros`
* [ ] `_mm_test_mix_ones_zeros`
* [ ] `_mm_test_all_ones`
* [ ] `_mm_round_pd`
* [ ] `_mm_floor_pd`
* [ ] `_mm_ceil_pd`
* [ ] `_mm_round_ps`
* [ ] `_mm_floor_ps`
* [ ] `_mm_ceil_ps`
* [ ] `_mm_round_sd`
* [ ] `_mm_floor_sd`
* [ ] `_mm_ceil_sd`
* [ ] `_mm_round_ss`
* [ ] `_mm_floor_ss`
* [ ] `_mm_ceil_ss`
* [ ] `_mm_minpos_epu16`
* [ ] `_mm_mpsadbw_epu8`
* [ ] `_mm_stream_load_si128`


sse4.2
------
* [ ] `_mm_cmpistrm`
* [ ] `_mm_cmpistri`
* [ ] `_mm_cmpistrz`
* [ ] `_mm_cmpistrc`
* [ ] `_mm_cmpistrs`
* [ ] `_mm_cmpistro`
* [ ] `_mm_cmpistra`
* [ ] `_mm_cmpestrm`
* [ ] `_mm_cmpestri`
* [ ] `_mm_cmpestrz`
* [ ] `_mm_cmpestrc`
* [ ] `_mm_cmpestrs`
* [ ] `_mm_cmpestro`
* [ ] `_mm_cmpestra`
* [ ] `_mm_cmpgt_epi64`
* [ ] `_mm_crc32_u8`
* [ ] `_mm_crc32_u16`
* [ ] `_mm_crc32_u32`
* [ ] `_mm_crc32_u64`


avx
---
* [x] `_mm256_add_pd`
* [x] `_mm256_add_ps`
* [x] `_mm256_addsub_pd`
* [ ] `_mm256_addsub_ps`
* [ ] `_mm256_and_pd`
* [ ] `_mm256_and_ps`
* [ ] `_mm256_andnot_pd`
* [ ] `_mm256_andnot_ps`
* [ ] `_mm256_blend_pd`
* [ ] `_mm256_blend_ps`
* [ ] `_mm256_blendv_pd`
* [ ] `_mm256_blendv_ps`
* [ ] `_mm256_div_pd`
* [ ] `_mm256_div_ps`
* [ ] `_mm256_dp_ps`
* [ ] `_mm256_hadd_pd`
* [ ] `_mm256_hadd_ps`
* [ ] `_mm256_hsub_pd`
* [ ] `_mm256_hsub_ps`
* [ ] `_mm256_max_pd`
* [ ] `_mm256_max_ps`
* [ ] `_mm256_min_pd`
* [ ] `_mm256_min_ps`
* [ ] `_mm256_mul_pd`
* [ ] `_mm256_mul_ps`
* [ ] `_mm256_or_pd`
* [ ] `_mm256_or_ps`
* [ ] `_mm256_shuffle_pd`
* [ ] `_mm256_shuffle_ps`
* [ ] `_mm256_sub_pd`
* [ ] `_mm256_sub_ps`
* [ ] `_mm256_xor_pd`
* [ ] `_mm256_xor_ps`
* [ ] `_mm_cmp_pd`
* [ ] `_mm256_cmp_pd`
* [ ] `_mm_cmp_ps`
* [ ] `_mm256_cmp_ps`
* [ ] `_mm_cmp_sd`
* [ ] `_mm_cmp_ss`
* [ ] `_mm256_cvtepi32_pd`
* [ ] `_mm256_cvtepi32_ps`
* [ ] `_mm256_cvtpd_ps`
* [ ] `_mm256_cvtps_epi32`
* [ ] `_mm256_cvtps_pd`
* [ ] `_mm256_cvttpd_epi32`
* [ ] `_mm256_cvtpd_epi32`
* [ ] `_mm256_cvttps_epi32`
* [ ] `_mm256_extractf128_ps`
* [ ] `_mm256_extractf128_pd`
* [ ] `_mm256_extractf128_si256`
* [ ] `_mm256_extract_epi8`
* [ ] `_mm256_extract_epi16`
* [ ] `_mm256_extract_epi32`
* [ ] `_mm256_extract_epi64`
* [ ] `_mm256_zeroall`
* [ ] `_mm256_zeroupper`
* [ ] `_mm256_permutevar_ps`
* [ ] `_mm_permutevar_ps`
* [ ] `_mm256_permute_ps`
* [ ] `_mm_permute_ps`
* [ ] `_mm256_permutevar_pd`
* [ ] `_mm_permutevar_pd`
* [ ] `_mm256_permute_pd`
* [ ] `_mm_permute_pd`
* [ ] `_mm256_permute2f128_ps`
* [ ] `_mm256_permute2f128_pd`
* [ ] `_mm256_permute2f128_si256`
* [ ] `_mm256_broadcast_ss`
* [ ] `_mm_broadcast_ss`
* [ ] `_mm256_broadcast_sd`
* [ ] `_mm256_broadcast_ps`
* [ ] `_mm256_broadcast_pd`
* [ ] `_mm256_insertf128_ps`
* [ ] `_mm256_insertf128_pd`
* [ ] `_mm256_insertf128_si256`
* [ ] `_mm256_insert_epi8`
* [ ] `_mm256_insert_epi16`
* [ ] `_mm256_insert_epi32`
* [ ] `_mm256_insert_epi64`
* [ ] `_mm256_load_pd`
* [ ] `_mm256_store_pd`
* [ ] `_mm256_load_ps`
* [ ] `_mm256_store_ps`
* [ ] `_mm256_loadu_pd`
* [ ] `_mm256_storeu_pd`
* [ ] `_mm256_loadu_ps`
* [ ] `_mm256_storeu_ps`
* [ ] `_mm256_load_si256`
* [ ] `_mm256_store_si256`
* [ ] `_mm256_loadu_si256`
* [ ] `_mm256_storeu_si256`
* [ ] `_mm256_maskload_pd`
* [ ] `_mm256_maskstore_pd`
* [ ] `_mm_maskload_pd`
* [ ] `_mm_maskstore_pd`
* [ ] `_mm256_maskload_ps`
* [ ] `_mm256_maskstore_ps`
* [ ] `_mm_maskload_ps`
* [ ] `_mm_maskstore_ps`
* [ ] `_mm256_movehdup_ps`
* [ ] `_mm256_moveldup_ps`
* [ ] `_mm256_movedup_pd`
* [ ] `_mm256_lddqu_si256`
* [ ] `_mm256_stream_si256`
* [ ] `_mm256_stream_pd`
* [ ] `_mm256_stream_ps`
* [ ] `_mm256_rcp_ps`
* [ ] `_mm256_rsqrt_ps`
* [ ] `_mm256_sqrt_pd`
* [ ] `_mm256_sqrt_ps`
* [ ] `_mm256_round_pd`
* [ ] `_mm256_round_ps`
* [ ] `_mm256_unpackhi_pd`
* [ ] `_mm256_unpackhi_ps`
* [ ] `_mm256_unpacklo_pd`
* [ ] `_mm256_unpacklo_ps`
* [ ] `_mm256_testz_si256`
* [ ] `_mm256_testc_si256`
* [ ] `_mm256_testnzc_si256`
* [ ] `_mm256_testz_pd`
* [ ] `_mm256_testc_pd`
* [ ] `_mm256_testnzc_pd`
* [ ] `_mm_testz_pd`
* [ ] `_mm_testc_pd`
* [ ] `_mm_testnzc_pd`
* [ ] `_mm256_testz_ps`
* [ ] `_mm256_testc_ps`
* [ ] `_mm256_testnzc_ps`
* [ ] `_mm_testz_ps`
* [ ] `_mm_testc_ps`
* [ ] `_mm_testnzc_ps`
* [ ] `_mm256_movemask_pd`
* [ ] `_mm256_movemask_ps`
* [ ] `_mm256_setzero_pd`
* [ ] `_mm256_setzero_ps`
* [ ] `_mm256_setzero_si256`
* [ ] `_mm256_set_pd`
* [ ] `_mm256_set_ps`
* [ ] `_mm256_set_epi8`
* [ ] `_mm256_set_epi16`
* [ ] `_mm256_set_epi32`
* [ ] `_mm256_set_epi64x`
* [ ] `_mm256_setr_pd`
* [ ] `_mm256_setr_ps`
* [ ] `_mm256_setr_epi8`
* [ ] `_mm256_setr_epi16`
* [ ] `_mm256_setr_epi32`
* [ ] `_mm256_setr_epi64x`
* [ ] `_mm256_set1_pd`
* [ ] `_mm256_set1_ps`
* [ ] `_mm256_set1_epi8`
* [ ] `_mm256_set1_epi16`
* [ ] `_mm256_set1_epi32`
* [ ] `_mm256_set1_epi64x`
* [ ] `_mm256_castpd_ps`
* [ ] `_mm256_castps_pd`
* [ ] `_mm256_castps_si256`
* [ ] `_mm256_castpd_si256`
* [ ] `_mm256_castsi256_ps`
* [ ] `_mm256_castsi256_pd`
* [ ] `_mm256_castps256_ps128`
* [ ] `_mm256_castpd256_pd128`
* [ ] `_mm256_castsi256_si128`
* [ ] `_mm256_castps128_ps256`
* [ ] `_mm256_castpd128_pd256`
* [ ] `_mm256_castsi128_si256`
* [ ] `_mm256_zextps128_ps256`
* [ ] `_mm256_zextpd128_pd256`
* [ ] `_mm256_zextsi128_si256`
* [ ] `_mm256_floor_ps`
* [ ] `_mm256_ceil_ps`
* [ ] `_mm256_floor_pd`
* [ ] `_mm256_ceil_pd`
* [ ] `_mm256_undefined_ps`
* [ ] `_mm256_undefined_pd`
* [ ] `_mm256_undefined_si256`
* [ ] `_mm256_set_m128`
* [ ] `_mm256_set_m128d`
* [ ] `_mm256_set_m128i`
* [ ] `_mm256_setr_m128`
* [ ] `_mm256_setr_m128d`
* [ ] `_mm256_setr_m128i`
* [ ] `_mm256_loadu2_m128`
* [ ] `_mm256_loadu2_m128d`
* [ ] `_mm256_loadu2_m128i`
* [ ] `_mm256_storeu2_m128`
* [ ] `_mm256_storeu2_m128d`
* [ ] `_mm256_storeu2_m128i`



avx2
----
* [x] `_mm256_abs_epi8`
* [x] `_mm256_abs_epi16`
* [x] `_mm256_abs_epi32`
* [x] `_mm256_add_epi8`
* [x] `_mm256_add_epi16`
* [x] `_mm256_add_epi32`
* [x] `_mm256_add_epi64`
* [x] `_mm256_adds_epi8`
* [x] `_mm256_adds_epi16`
* [x] `_mm256_adds_epu8`
* [x] `_mm256_adds_epu16`
* [ ] `_mm256_alignr_epi8`
* [x] `_mm256_and_si256`
* [x] `_mm256_andnot_si256`
* [x] `_mm256_avg_epu8`
* [x] `_mm256_avg_epu16`
* [ ] `_mm256_blend_epi16`
* [ ] `_mm_blend_epi32`
* [ ] `_mm256_blend_epi32`
* [x] `_mm256_blendv_epi8`
* [ ] `_mm_broadcastb_epi8`
* [ ] `_mm256_broadcastb_epi8`
* [ ] `_mm_broadcastd_epi32`
* [ ] `_mm256_broadcastd_epi32`
* [ ] `_mm_broadcastq_epi64`
* [ ] `_mm256_broadcastq_epi64`
* [ ] `_mm_broadcastsd_pd`
* [ ] `_mm256_broadcastsd_pd`
* [ ] `_mm_broadcastsi128_si256`
* [ ] `_mm256_broadcastsi128_si256`
* [ ] `_mm_broadcastss_ps`
* [ ] `_mm256_broadcastss_ps`
* [ ] `_mm_broadcastw_epi16`
* [ ] `_mm256_broadcastw_epi16`
* [x] `_mm256_cmpeq_epi8`
* [x] `_mm256_cmpeq_epi16`
* [x] `_mm256_cmpeq_epi32`
* [x] `_mm256_cmpeq_epi64`
* [x] `_mm256_cmpgt_epi8`
* [x] `_mm256_cmpgt_epi16`
* [x] `_mm256_cmpgt_epi32`
* [x] `_mm256_cmpgt_epi64`
* [ ] `_mm256_cvtepi16_epi32`
* [ ] `_mm256_cvtepi16_epi64`
* [ ] `_mm256_cvtepi32_epi64`
* [ ] `_mm256_cvtepi8_epi16`
* [ ] `_mm256_cvtepi8_epi32`
* [ ] `_mm256_cvtepi8_epi64`
* [ ] `_mm256_cvtepu16_epi32`
* [ ] `_mm256_cvtepu16_epi64`
* [ ] `_mm256_cvtepu32_epi64`
* [ ] `_mm256_cvtepu8_epi16`
* [ ] `_mm256_cvtepu8_epi32`
* [ ] `_mm256_cvtepu8_epi64`
* [ ] `_mm256_extracti128_si256`
* [x] `_mm256_hadd_epi16`
* [x] `_mm256_hadd_epi32`
* [x] `_mm256_hadds_epi16`
* [x] `_mm256_hsub_epi16`
* [x] `_mm256_hsub_epi32`
* [x] `_mm256_hsubs_epi16`
* [ ] `_mm_i32gather_pd`
* [ ] `_mm256_i32gather_pd`
* [ ] `_mm_i32gather_ps`
* [ ] `_mm256_i32gather_ps`
* [ ] `_mm_i32gather_epi32`
* [ ] `_mm256_i32gather_epi32`
* [ ] `_mm_i32gather_epi64`
* [ ] `_mm256_i32gather_epi64`
* [ ] `_mm_i64gather_pd`
* [ ] `_mm256_i64gather_pd`
* [ ] `_mm_i64gather_ps`
* [ ] `_mm256_i64gather_ps`
* [ ] `_mm_i64gather_epi32`
* [ ] `_mm256_i64gather_epi32`
* [ ] `_mm_i64gather_epi64`
* [ ] `_mm256_i64gather_epi64`
* [ ] `_mm256_inserti128_si256`
* [ ] `_mm256_madd_epi16`
* [ ] `_mm256_maddubs_epi16`
* [ ] `_mm_mask_i32gather_pd`
* [ ] `_mm256_mask_i32gather_pd`
* [ ] `_mm_mask_i32gather_ps`
* [ ] `_mm256_mask_i32gather_ps`
* [ ] `_mm_mask_i32gather_epi32`
* [ ] `_mm256_mask_i32gather_epi32`
* [ ] `_mm_mask_i32gather_epi64`
* [ ] `_mm256_mask_i32gather_epi64`
* [ ] `_mm_mask_i64gather_pd`
* [ ] `_mm256_mask_i64gather_pd`
* [ ] `_mm_mask_i64gather_ps`
* [ ] `_mm256_mask_i64gather_ps`
* [ ] `_mm_mask_i64gather_epi32`
* [ ] `_mm256_mask_i64gather_epi32`
* [ ] `_mm_mask_i64gather_epi64`
* [ ] `_mm256_mask_i64gather_epi64`
* [ ] `_mm_maskload_epi32`
* [ ] `_mm256_maskload_epi32`
* [ ] `_mm_maskload_epi64`
* [ ] `_mm256_maskload_epi64`
* [ ] `_mm_maskstore_epi32`
* [ ] `_mm256_maskstore_epi32`
* [ ] `_mm_maskstore_epi64`
* [ ] `_mm256_maskstore_epi64`
* [ ] `_mm256_max_epi8`
* [ ] `_mm256_max_epi16`
* [ ] `_mm256_max_epi32`
* [ ] `_mm256_max_epu8`
* [ ] `_mm256_max_epu16`
* [ ] `_mm256_max_epu32`
* [ ] `_mm256_min_epi8`
* [ ] `_mm256_min_epi16`
* [ ] `_mm256_min_epi32`
* [ ] `_mm256_min_epu8`
* [ ] `_mm256_min_epu16`
* [ ] `_mm256_min_epu32`
* [ ] `_mm256_movemask_epi8`
* [ ] `_mm256_mpsadbw_epu8`
* [ ] `_mm256_mul_epi32`
* [ ] `_mm256_mul_epu32`
* [ ] `_mm256_mulhi_epi16`
* [ ] `_mm256_mulhi_epu16`
* [ ] `_mm256_mulhrs_epi16`
* [ ] `_mm256_mullo_epi16`
* [ ] `_mm256_mullo_epi32`
* [ ] `_mm256_or_si256`
* [ ] `_mm256_packs_epi16`
* [ ] `_mm256_packs_epi32`
* [ ] `_mm256_packus_epi16`
* [ ] `_mm256_packus_epi32`
* [ ] `_mm256_permute2x128_si256`
* [ ] `_mm256_permute4x64_epi64`
* [ ] `_mm256_permute4x64_pd`
* [ ] `_mm256_permutevar8x32_epi32`
* [ ] `_mm256_permutevar8x32_ps`
* [ ] `_mm256_sad_epu8`
* [ ] `_mm256_shuffle_epi32`
* [ ] `_mm256_shuffle_epi8`
* [ ] `_mm256_shufflehi_epi16`
* [ ] `_mm256_shufflelo_epi16`
* [ ] `_mm256_sign_epi8`
* [ ] `_mm256_sign_epi16`
* [ ] `_mm256_sign_epi32`
* [ ] `_mm256_slli_si256`
* [ ] `_mm256_bslli_epi128`
* [ ] `_mm256_sll_epi16`
* [ ] `_mm256_slli_epi16`
* [ ] `_mm256_sll_epi32`
* [ ] `_mm256_slli_epi32`
* [ ] `_mm256_sll_epi64`
* [ ] `_mm256_slli_epi64`
* [ ] `_mm_sllv_epi32`
* [ ] `_mm256_sllv_epi32`
* [ ] `_mm_sllv_epi64`
* [ ] `_mm256_sllv_epi64`
* [ ] `_mm256_sra_epi16`
* [ ] `_mm256_srai_epi16`
* [ ] `_mm256_sra_epi32`
* [ ] `_mm256_srai_epi32`
* [ ] `_mm_srav_epi32`
* [ ] `_mm256_srav_epi32`
* [ ] `_mm256_srli_si256`
* [ ] `_mm256_bsrli_epi128`
* [ ] `_mm256_srl_epi16`
* [ ] `_mm256_srli_epi16`
* [ ] `_mm256_srl_epi32`
* [ ] `_mm256_srli_epi32`
* [ ] `_mm256_srl_epi64`
* [ ] `_mm256_srli_epi64`
* [ ] `_mm_srlv_epi32`
* [ ] `_mm256_srlv_epi32`
* [ ] `_mm_srlv_epi64`
* [ ] `_mm256_srlv_epi64`
* [ ] `_mm256_stream_load_si256`
* [ ] `_mm256_sub_epi8`
* [ ] `_mm256_sub_epi16`
* [ ] `_mm256_sub_epi32`
* [ ] `_mm256_sub_epi64`
* [ ] `_mm256_subs_epi8`
* [ ] `_mm256_subs_epi16`
* [ ] `_mm256_subs_epu8`
* [ ] `_mm256_subs_epu16`
* [ ] `_mm256_xor_si256`
* [ ] `_mm256_unpackhi_epi8`
* [ ] `_mm256_unpackhi_epi16`
* [ ] `_mm256_unpackhi_epi32`
* [ ] `_mm256_unpackhi_epi64`
* [ ] `_mm256_unpacklo_epi8`
* [ ] `_mm256_unpacklo_epi16`
* [ ] `_mm256_unpacklo_epi32`
* [ ] `_mm256_unpacklo_epi64`
