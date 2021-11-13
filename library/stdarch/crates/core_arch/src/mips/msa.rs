//! MIPS SIMD Architecture intrinsics
//!
//! The reference is [MIPS Architecture for Programmers Volume IV-j: The
//! MIPS32 SIMD Architecture Module Revision 1.12][msa_ref].
//!
//! [msa_ref]: http://cdn2.imgtec.com/documentation/MD00866-2B-MSA32-AFP-01.12.pdf

#[cfg(test)]
use stdarch_test::assert_instr;

use crate::mem;

#[macro_use]
mod macros;

types! {
    // / MIPS-specific 128-bit wide vector of 16 packed `i8`.
   pub struct v16i8(
       i8, i8, i8, i8, i8, i8, i8, i8,
       i8, i8, i8, i8, i8, i8, i8, i8,
   );

    // / MIPS-specific 128-bit wide vector of 8 packed `i16`.
   pub struct v8i16(
       i16, i16, i16, i16, i16, i16, i16, i16,
   );

    // / MIPS-specific 128-bit wide vector of 4 packed `i32`.
   pub struct v4i32(
       i32, i32, i32, i32,
   );

    // / MIPS-specific 128-bit wide vector of 2 packed `i64`.
   pub struct v2i64(
       i64, i64,
   );

    // / MIPS-specific 128-bit wide vector of 16 packed `u8`.
   pub struct v16u8(
       u8, u8, u8, u8, u8, u8, u8, u8,
       u8, u8, u8, u8, u8, u8, u8, u8,
   );

    // / MIPS-specific 128-bit wide vector of 8 packed `u16`.
   pub struct v8u16(
       u16, u16, u16, u16, u16, u16, u16, u16,
   );

    // / MIPS-specific 128-bit wide vector of 4 packed `u32`.
   pub struct v4u32(
       u32, u32, u32, u32,
   );

    // / MIPS-specific 128-bit wide vector of 2 packed `u64`.
   pub struct v2u64(
       u64, u64,
   );

   // / MIPS-specific 128-bit wide vector of 4 packed `f32`.
   pub struct v4f32(
       f32, f32, f32, f32,
   );

    // / MIPS-specific 128-bit wide vector of 2 packed `f64`.
   pub struct v2f64(
       f64, f64,
   );
}

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.mips.add.a.b"]
    fn msa_add_a_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.mips.add.a.h"]
    fn msa_add_a_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.mips.add.a.w"]
    fn msa_add_a_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.mips.add.a.d"]
    fn msa_add_a_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.mips.adds.a.b"]
    fn msa_adds_a_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.mips.adds.a.h"]
    fn msa_adds_a_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.mips.adds.a.w"]
    fn msa_adds_a_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.mips.adds.a.d"]
    fn msa_adds_a_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.mips.adds.s.b"]
    fn msa_adds_s_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.mips.adds.s.h"]
    fn msa_adds_s_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.mips.adds.s.w"]
    fn msa_adds_s_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.mips.adds.s.d"]
    fn msa_adds_s_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.mips.adds.u.b"]
    fn msa_adds_u_b(a: v16u8, b: v16u8) -> v16u8;
    #[link_name = "llvm.mips.adds.u.h"]
    fn msa_adds_u_h(a: v8u16, b: v8u16) -> v8u16;
    #[link_name = "llvm.mips.adds.u.w"]
    fn msa_adds_u_w(a: v4u32, b: v4u32) -> v4u32;
    #[link_name = "llvm.mips.adds.u.d"]
    fn msa_adds_u_d(a: v2u64, b: v2u64) -> v2u64;
    #[link_name = "llvm.mips.addv.b"]
    fn msa_addv_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.mips.addv.h"]
    fn msa_addv_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.mips.addv.w"]
    fn msa_addv_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.mips.addv.d"]
    fn msa_addv_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.mips.addvi.b"]
    fn msa_addvi_b(a: v16i8, b: i32) -> v16i8;
    #[link_name = "llvm.mips.addvi.h"]
    fn msa_addvi_h(a: v8i16, b: i32) -> v8i16;
    #[link_name = "llvm.mips.addvi.w"]
    fn msa_addvi_w(a: v4i32, b: i32) -> v4i32;
    #[link_name = "llvm.mips.addvi.d"]
    fn msa_addvi_d(a: v2i64, b: i32) -> v2i64;
    #[link_name = "llvm.mips.and.v"]
    fn msa_and_v(a: v16u8, b: v16u8) -> v16u8;
    #[link_name = "llvm.mips.andi.b"]
    fn msa_andi_b(a: v16u8, b: i32) -> v16u8;
    #[link_name = "llvm.mips.asub.s.b"]
    fn msa_asub_s_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.mips.asub.s.h"]
    fn msa_asub_s_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.mips.asub.s.w"]
    fn msa_asub_s_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.mips.asub.s.d"]
    fn msa_asub_s_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.mips.asub.u.b"]
    fn msa_asub_u_b(a: v16u8, b: v16u8) -> v16u8;
    #[link_name = "llvm.mips.asub.u.h"]
    fn msa_asub_u_h(a: v8u16, b: v8u16) -> v8u16;
    #[link_name = "llvm.mips.asub.u.w"]
    fn msa_asub_u_w(a: v4u32, b: v4u32) -> v4u32;
    #[link_name = "llvm.mips.asub.u.d"]
    fn msa_asub_u_d(a: v2u64, b: v2u64) -> v2u64;
    #[link_name = "llvm.mips.ave.s.b"]
    fn msa_ave_s_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.mips.ave.s.h"]
    fn msa_ave_s_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.mips.ave.s.w"]
    fn msa_ave_s_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.mips.ave.s.d"]
    fn msa_ave_s_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.mips.ave.u.b"]
    fn msa_ave_u_b(a: v16u8, b: v16u8) -> v16u8;
    #[link_name = "llvm.mips.ave.u.h"]
    fn msa_ave_u_h(a: v8u16, b: v8u16) -> v8u16;
    #[link_name = "llvm.mips.ave.u.w"]
    fn msa_ave_u_w(a: v4u32, b: v4u32) -> v4u32;
    #[link_name = "llvm.mips.ave.u.d"]
    fn msa_ave_u_d(a: v2u64, b: v2u64) -> v2u64;
    #[link_name = "llvm.mips.aver.s.b"]
    fn msa_aver_s_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.mips.aver.s.h"]
    fn msa_aver_s_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.mips.aver.s.w"]
    fn msa_aver_s_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.mips.aver.s.d"]
    fn msa_aver_s_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.mips.aver.u.b"]
    fn msa_aver_u_b(a: v16u8, b: v16u8) -> v16u8;
    #[link_name = "llvm.mips.aver.u.h"]
    fn msa_aver_u_h(a: v8u16, b: v8u16) -> v8u16;
    #[link_name = "llvm.mips.aver.u.w"]
    fn msa_aver_u_w(a: v4u32, b: v4u32) -> v4u32;
    #[link_name = "llvm.mips.aver.u.d"]
    fn msa_aver_u_d(a: v2u64, b: v2u64) -> v2u64;
    #[link_name = "llvm.mips.bclr.b"]
    fn msa_bclr_b(a: v16u8, b: v16u8) -> v16u8;
    #[link_name = "llvm.mips.bclr.h"]
    fn msa_bclr_h(a: v8u16, b: v8u16) -> v8u16;
    #[link_name = "llvm.mips.bclr.w"]
    fn msa_bclr_w(a: v4u32, b: v4u32) -> v4u32;
    #[link_name = "llvm.mips.bclr.d"]
    fn msa_bclr_d(a: v2u64, b: v2u64) -> v2u64;
    #[link_name = "llvm.mips.bclri.b"]
    fn msa_bclri_b(a: v16u8, b: i32) -> v16u8;
    #[link_name = "llvm.mips.bclri.h"]
    fn msa_bclri_h(a: v8u16, b: i32) -> v8u16;
    #[link_name = "llvm.mips.bclri.w"]
    fn msa_bclri_w(a: v4u32, b: i32) -> v4u32;
    #[link_name = "llvm.mips.bclri.d"]
    fn msa_bclri_d(a: v2u64, b: i32) -> v2u64;
    #[link_name = "llvm.mips.binsl.b"]
    fn msa_binsl_b(a: v16u8, b: v16u8, c: v16u8) -> v16u8;
    #[link_name = "llvm.mips.binsl.h"]
    fn msa_binsl_h(a: v8u16, b: v8u16, c: v8u16) -> v8u16;
    #[link_name = "llvm.mips.binsl.w"]
    fn msa_binsl_w(a: v4u32, b: v4u32, c: v4u32) -> v4u32;
    #[link_name = "llvm.mips.binsl.d"]
    fn msa_binsl_d(a: v2u64, b: v2u64, c: v2u64) -> v2u64;
    #[link_name = "llvm.mips.binsli.b"]
    fn msa_binsli_b(a: v16u8, b: v16u8, c: i32) -> v16u8;
    #[link_name = "llvm.mips.binsli.h"]
    fn msa_binsli_h(a: v8u16, b: v8u16, c: i32) -> v8u16;
    #[link_name = "llvm.mips.binsli.w"]
    fn msa_binsli_w(a: v4u32, b: v4u32, c: i32) -> v4u32;
    #[link_name = "llvm.mips.binsli.d"]
    fn msa_binsli_d(a: v2u64, b: v2u64, c: i32) -> v2u64;
    #[link_name = "llvm.mips.binsr.b"]
    fn msa_binsr_b(a: v16u8, b: v16u8, c: v16u8) -> v16u8;
    #[link_name = "llvm.mips.binsr.h"]
    fn msa_binsr_h(a: v8u16, b: v8u16, c: v8u16) -> v8u16;
    #[link_name = "llvm.mips.binsr.w"]
    fn msa_binsr_w(a: v4u32, b: v4u32, c: v4u32) -> v4u32;
    #[link_name = "llvm.mips.binsr.d"]
    fn msa_binsr_d(a: v2u64, b: v2u64, c: v2u64) -> v2u64;
    #[link_name = "llvm.mips.binsri.b"]
    fn msa_binsri_b(a: v16u8, b: v16u8, c: i32) -> v16u8;
    #[link_name = "llvm.mips.binsri.h"]
    fn msa_binsri_h(a: v8u16, b: v8u16, c: i32) -> v8u16;
    #[link_name = "llvm.mips.binsri.w"]
    fn msa_binsri_w(a: v4u32, b: v4u32, c: i32) -> v4u32;
    #[link_name = "llvm.mips.binsri.d"]
    fn msa_binsri_d(a: v2u64, b: v2u64, c: i32) -> v2u64;
    #[link_name = "llvm.mips.bmnz.v"]
    fn msa_bmnz_v(a: v16u8, b: v16u8, c: v16u8) -> v16u8;
    #[link_name = "llvm.mips.bmnzi.b"]
    fn msa_bmnzi_b(a: v16u8, b: v16u8, c: i32) -> v16u8;
    #[link_name = "llvm.mips.bmz.v"]
    fn msa_bmz_v(a: v16u8, b: v16u8, c: v16u8) -> v16u8;
    #[link_name = "llvm.mips.bmzi.b"]
    fn msa_bmzi_b(a: v16u8, b: v16u8, c: i32) -> v16u8;
    #[link_name = "llvm.mips.bneg.b"]
    fn msa_bneg_b(a: v16u8, b: v16u8) -> v16u8;
    #[link_name = "llvm.mips.bneg.h"]
    fn msa_bneg_h(a: v8u16, b: v8u16) -> v8u16;
    #[link_name = "llvm.mips.bneg.w"]
    fn msa_bneg_w(a: v4u32, b: v4u32) -> v4u32;
    #[link_name = "llvm.mips.bneg.d"]
    fn msa_bneg_d(a: v2u64, b: v2u64) -> v2u64;
    #[link_name = "llvm.mips.bnegi.b"]
    fn msa_bnegi_b(a: v16u8, b: i32) -> v16u8;
    #[link_name = "llvm.mips.bnegi.h"]
    fn msa_bnegi_h(a: v8u16, b: i32) -> v8u16;
    #[link_name = "llvm.mips.bnegi.w"]
    fn msa_bnegi_w(a: v4u32, b: i32) -> v4u32;
    #[link_name = "llvm.mips.bnegi.d"]
    fn msa_bnegi_d(a: v2u64, b: i32) -> v2u64;
    #[link_name = "llvm.mips.bnz.b"]
    fn msa_bnz_b(a: v16u8) -> i32;
    #[link_name = "llvm.mips.bnz.h"]
    fn msa_bnz_h(a: v8u16) -> i32;
    #[link_name = "llvm.mips.bnz.w"]
    fn msa_bnz_w(a: v4u32) -> i32;
    #[link_name = "llvm.mips.bnz.d"]
    fn msa_bnz_d(a: v2u64) -> i32;
    #[link_name = "llvm.mips.bnz.v"]
    fn msa_bnz_v(a: v16u8) -> i32;
    #[link_name = "llvm.mips.bsel.v"]
    fn msa_bsel_v(a: v16u8, b: v16u8, c: v16u8) -> v16u8;
    #[link_name = "llvm.mips.bseli.b"]
    fn msa_bseli_b(a: v16u8, b: v16u8, c: i32) -> v16u8;
    #[link_name = "llvm.mips.bset.b"]
    fn msa_bset_b(a: v16u8, b: v16u8) -> v16u8;
    #[link_name = "llvm.mips.bset.h"]
    fn msa_bset_h(a: v8u16, b: v8u16) -> v8u16;
    #[link_name = "llvm.mips.bset.w"]
    fn msa_bset_w(a: v4u32, b: v4u32) -> v4u32;
    #[link_name = "llvm.mips.bset.d"]
    fn msa_bset_d(a: v2u64, b: v2u64) -> v2u64;
    #[link_name = "llvm.mips.bseti.b"]
    fn msa_bseti_b(a: v16u8, b: i32) -> v16u8;
    #[link_name = "llvm.mips.bseti.h"]
    fn msa_bseti_h(a: v8u16, b: i32) -> v8u16;
    #[link_name = "llvm.mips.bseti.w"]
    fn msa_bseti_w(a: v4u32, b: i32) -> v4u32;
    #[link_name = "llvm.mips.bseti.d"]
    fn msa_bseti_d(a: v2u64, b: i32) -> v2u64;
    #[link_name = "llvm.mips.bz.b"]
    fn msa_bz_b(a: v16u8) -> i32;
    #[link_name = "llvm.mips.bz.h"]
    fn msa_bz_h(a: v8u16) -> i32;
    #[link_name = "llvm.mips.bz.w"]
    fn msa_bz_w(a: v4u32) -> i32;
    #[link_name = "llvm.mips.bz.d"]
    fn msa_bz_d(a: v2u64) -> i32;
    #[link_name = "llvm.mips.bz.v"]
    fn msa_bz_v(a: v16u8) -> i32;
    #[link_name = "llvm.mips.ceq.b"]
    fn msa_ceq_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.mips.ceq.h"]
    fn msa_ceq_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.mips.ceq.w"]
    fn msa_ceq_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.mips.ceq.d"]
    fn msa_ceq_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.mips.ceqi.b"]
    fn msa_ceqi_b(a: v16i8, b: i32) -> v16i8;
    #[link_name = "llvm.mips.ceqi.h"]
    fn msa_ceqi_h(a: v8i16, b: i32) -> v8i16;
    #[link_name = "llvm.mips.ceqi.w"]
    fn msa_ceqi_w(a: v4i32, b: i32) -> v4i32;
    #[link_name = "llvm.mips.ceqi.d"]
    fn msa_ceqi_d(a: v2i64, b: i32) -> v2i64;
    #[link_name = "llvm.mips.cfcmsa"]
    fn msa_cfcmsa(a: i32) -> i32;
    #[link_name = "llvm.mips.cle.s.b"]
    fn msa_cle_s_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.mips.cle.s.h"]
    fn msa_cle_s_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.mips.cle.s.w"]
    fn msa_cle_s_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.mips.cle.s.d"]
    fn msa_cle_s_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.mips.cle.u.b"]
    fn msa_cle_u_b(a: v16u8, b: v16u8) -> v16i8;
    #[link_name = "llvm.mips.cle.u.h"]
    fn msa_cle_u_h(a: v8u16, b: v8u16) -> v8i16;
    #[link_name = "llvm.mips.cle.u.w"]
    fn msa_cle_u_w(a: v4u32, b: v4u32) -> v4i32;
    #[link_name = "llvm.mips.cle.u.d"]
    fn msa_cle_u_d(a: v2u64, b: v2u64) -> v2i64;
    #[link_name = "llvm.mips.clei.s.b"]
    fn msa_clei_s_b(a: v16i8, b: i32) -> v16i8;
    #[link_name = "llvm.mips.clei.s.h"]
    fn msa_clei_s_h(a: v8i16, b: i32) -> v8i16;
    #[link_name = "llvm.mips.clei.s.w"]
    fn msa_clei_s_w(a: v4i32, b: i32) -> v4i32;
    #[link_name = "llvm.mips.clei.s.d"]
    fn msa_clei_s_d(a: v2i64, b: i32) -> v2i64;
    #[link_name = "llvm.mips.clei.u.b"]
    fn msa_clei_u_b(a: v16u8, b: i32) -> v16i8;
    #[link_name = "llvm.mips.clei.u.h"]
    fn msa_clei_u_h(a: v8u16, b: i32) -> v8i16;
    #[link_name = "llvm.mips.clei.u.w"]
    fn msa_clei_u_w(a: v4u32, b: i32) -> v4i32;
    #[link_name = "llvm.mips.clei.u.d"]
    fn msa_clei_u_d(a: v2u64, b: i32) -> v2i64;
    #[link_name = "llvm.mips.clt.s.b"]
    fn msa_clt_s_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.mips.clt.s.h"]
    fn msa_clt_s_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.mips.clt.s.w"]
    fn msa_clt_s_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.mips.clt.s.d"]
    fn msa_clt_s_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.mips.clt.u.b"]
    fn msa_clt_u_b(a: v16u8, b: v16u8) -> v16i8;
    #[link_name = "llvm.mips.clt.u.h"]
    fn msa_clt_u_h(a: v8u16, b: v8u16) -> v8i16;
    #[link_name = "llvm.mips.clt.u.w"]
    fn msa_clt_u_w(a: v4u32, b: v4u32) -> v4i32;
    #[link_name = "llvm.mips.clt.u.d"]
    fn msa_clt_u_d(a: v2u64, b: v2u64) -> v2i64;
    #[link_name = "llvm.mips.clti.s.b"]
    fn msa_clti_s_b(a: v16i8, b: i32) -> v16i8;
    #[link_name = "llvm.mips.clti.s.h"]
    fn msa_clti_s_h(a: v8i16, b: i32) -> v8i16;
    #[link_name = "llvm.mips.clti.s.w"]
    fn msa_clti_s_w(a: v4i32, b: i32) -> v4i32;
    #[link_name = "llvm.mips.clti.s.d"]
    fn msa_clti_s_d(a: v2i64, b: i32) -> v2i64;
    #[link_name = "llvm.mips.clti.u.b"]
    fn msa_clti_u_b(a: v16u8, b: i32) -> v16i8;
    #[link_name = "llvm.mips.clti.u.h"]
    fn msa_clti_u_h(a: v8u16, b: i32) -> v8i16;
    #[link_name = "llvm.mips.clti.u.w"]
    fn msa_clti_u_w(a: v4u32, b: i32) -> v4i32;
    #[link_name = "llvm.mips.clti.u.d"]
    fn msa_clti_u_d(a: v2u64, b: i32) -> v2i64;
    #[link_name = "llvm.mips.copy.s.b"]
    fn msa_copy_s_b(a: v16i8, b: i32) -> i32;
    #[link_name = "llvm.mips.copy.s.h"]
    fn msa_copy_s_h(a: v8i16, b: i32) -> i32;
    #[link_name = "llvm.mips.copy.s.w"]
    fn msa_copy_s_w(a: v4i32, b: i32) -> i32;
    #[link_name = "llvm.mips.copy.s.d"]
    fn msa_copy_s_d(a: v2i64, b: i32) -> i64;
    #[link_name = "llvm.mips.copy.u.b"]
    fn msa_copy_u_b(a: v16i8, b: i32) -> u32;
    #[link_name = "llvm.mips.copy.u.h"]
    fn msa_copy_u_h(a: v8i16, b: i32) -> u32;
    #[link_name = "llvm.mips.copy.u.w"]
    fn msa_copy_u_w(a: v4i32, b: i32) -> u32;
    #[link_name = "llvm.mips.copy.u.d"]
    fn msa_copy_u_d(a: v2i64, b: i32) -> u64;
    #[link_name = "llvm.mips.ctcmsa"]
    fn msa_ctcmsa(imm5: i32, a: i32) -> ();
    #[link_name = "llvm.mips.div.s.b"]
    fn msa_div_s_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.mips.div.s.h"]
    fn msa_div_s_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.mips.div.s.w"]
    fn msa_div_s_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.mips.div.s.d"]
    fn msa_div_s_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.mips.div.u.b"]
    fn msa_div_u_b(a: v16u8, b: v16u8) -> v16u8;
    #[link_name = "llvm.mips.div.u.h"]
    fn msa_div_u_h(a: v8u16, b: v8u16) -> v8u16;
    #[link_name = "llvm.mips.div.u.w"]
    fn msa_div_u_w(a: v4u32, b: v4u32) -> v4u32;
    #[link_name = "llvm.mips.div.u.d"]
    fn msa_div_u_d(a: v2u64, b: v2u64) -> v2u64;
    #[link_name = "llvm.mips.dotp.s.h"]
    fn msa_dotp_s_h(a: v16i8, b: v16i8) -> v8i16;
    #[link_name = "llvm.mips.dotp.s.w"]
    fn msa_dotp_s_w(a: v8i16, b: v8i16) -> v4i32;
    #[link_name = "llvm.mips.dotp.s.d"]
    fn msa_dotp_s_d(a: v4i32, b: v4i32) -> v2i64;
    #[link_name = "llvm.mips.dotp.u.h"]
    fn msa_dotp_u_h(a: v16u8, b: v16u8) -> v8u16;
    #[link_name = "llvm.mips.dotp.u.w"]
    fn msa_dotp_u_w(a: v8u16, b: v8u16) -> v4u32;
    #[link_name = "llvm.mips.dotp.u.d"]
    fn msa_dotp_u_d(a: v4u32, b: v4u32) -> v2u64;
    #[link_name = "llvm.mips.dpadd.s.h"]
    fn msa_dpadd_s_h(a: v8i16, b: v16i8, c: v16i8) -> v8i16;
    #[link_name = "llvm.mips.dpadd.s.w"]
    fn msa_dpadd_s_w(a: v4i32, b: v8i16, c: v8i16) -> v4i32;
    #[link_name = "llvm.mips.dpadd.s.d"]
    fn msa_dpadd_s_d(a: v2i64, b: v4i32, c: v4i32) -> v2i64;
    #[link_name = "llvm.mips.dpadd.u.h"]
    fn msa_dpadd_u_h(a: v8u16, b: v16u8, c: v16u8) -> v8u16;
    #[link_name = "llvm.mips.dpadd.u.w"]
    fn msa_dpadd_u_w(a: v4u32, b: v8u16, c: v8u16) -> v4u32;
    #[link_name = "llvm.mips.dpadd.u.d"]
    fn msa_dpadd_u_d(a: v2u64, b: v4u32, c: v4u32) -> v2u64;
    #[link_name = "llvm.mips.dpsub.s.h"]
    fn msa_dpsub_s_h(a: v8i16, b: v16i8, c: v16i8) -> v8i16;
    #[link_name = "llvm.mips.dpsub.s.w"]
    fn msa_dpsub_s_w(a: v4i32, b: v8i16, c: v8i16) -> v4i32;
    #[link_name = "llvm.mips.dpsub.s.d"]
    fn msa_dpsub_s_d(a: v2i64, b: v4i32, c: v4i32) -> v2i64;
    #[link_name = "llvm.mips.dpsub.u.h"]
    fn msa_dpsub_u_h(a: v8i16, b: v16u8, c: v16u8) -> v8i16;
    #[link_name = "llvm.mips.dpsub.u.w"]
    fn msa_dpsub_u_w(a: v4i32, b: v8u16, c: v8u16) -> v4i32;
    #[link_name = "llvm.mips.dpsub.u.d"]
    fn msa_dpsub_u_d(a: v2i64, b: v4u32, c: v4u32) -> v2i64;
    #[link_name = "llvm.mips.fadd.w"]
    fn msa_fadd_w(a: v4f32, b: v4f32) -> v4f32;
    #[link_name = "llvm.mips.fadd.d"]
    fn msa_fadd_d(a: v2f64, b: v2f64) -> v2f64;
    #[link_name = "llvm.mips.fcaf.w"]
    fn msa_fcaf_w(a: v4f32, b: v4f32) -> v4i32;
    #[link_name = "llvm.mips.fcaf.d"]
    fn msa_fcaf_d(a: v2f64, b: v2f64) -> v2i64;
    #[link_name = "llvm.mips.fceq.w"]
    fn msa_fceq_w(a: v4f32, b: v4f32) -> v4i32;
    #[link_name = "llvm.mips.fceq.d"]
    fn msa_fceq_d(a: v2f64, b: v2f64) -> v2i64;
    #[link_name = "llvm.mips.fclass.w"]
    fn msa_fclass_w(a: v4f32) -> v4i32;
    #[link_name = "llvm.mips.fclass.d"]
    fn msa_fclass_d(a: v2f64) -> v2i64;
    #[link_name = "llvm.mips.fcle.w"]
    fn msa_fcle_w(a: v4f32, b: v4f32) -> v4i32;
    #[link_name = "llvm.mips.fcle.d"]
    fn msa_fcle_d(a: v2f64, b: v2f64) -> v2i64;
    #[link_name = "llvm.mips.fclt.w"]
    fn msa_fclt_w(a: v4f32, b: v4f32) -> v4i32;
    #[link_name = "llvm.mips.fclt.d"]
    fn msa_fclt_d(a: v2f64, b: v2f64) -> v2i64;
    #[link_name = "llvm.mips.fcne.w"]
    fn msa_fcne_w(a: v4f32, b: v4f32) -> v4i32;
    #[link_name = "llvm.mips.fcne.d"]
    fn msa_fcne_d(a: v2f64, b: v2f64) -> v2i64;
    #[link_name = "llvm.mips.fcor.w"]
    fn msa_fcor_w(a: v4f32, b: v4f32) -> v4i32;
    #[link_name = "llvm.mips.fcor.d"]
    fn msa_fcor_d(a: v2f64, b: v2f64) -> v2i64;
    #[link_name = "llvm.mips.fcueq.w"]
    fn msa_fcueq_w(a: v4f32, b: v4f32) -> v4i32;
    #[link_name = "llvm.mips.fcueq.d"]
    fn msa_fcueq_d(a: v2f64, b: v2f64) -> v2i64;
    #[link_name = "llvm.mips.fcule.w"]
    fn msa_fcule_w(a: v4f32, b: v4f32) -> v4i32;
    #[link_name = "llvm.mips.fcule.d"]
    fn msa_fcule_d(a: v2f64, b: v2f64) -> v2i64;
    #[link_name = "llvm.mips.fcult.w"]
    fn msa_fcult_w(a: v4f32, b: v4f32) -> v4i32;
    #[link_name = "llvm.mips.fcult.d"]
    fn msa_fcult_d(a: v2f64, b: v2f64) -> v2i64;
    #[link_name = "llvm.mips.fcun.w"]
    fn msa_fcun_w(a: v4f32, b: v4f32) -> v4i32;
    #[link_name = "llvm.mips.fcun.d"]
    fn msa_fcun_d(a: v2f64, b: v2f64) -> v2i64;
    #[link_name = "llvm.mips.fcune.w"]
    fn msa_fcune_w(a: v4f32, b: v4f32) -> v4i32;
    #[link_name = "llvm.mips.fcune.d"]
    fn msa_fcune_d(a: v2f64, b: v2f64) -> v2i64;
    #[link_name = "llvm.mips.fdiv.w"]
    fn msa_fdiv_w(a: v4f32, b: v4f32) -> v4f32;
    #[link_name = "llvm.mips.fdiv.d"]
    fn msa_fdiv_d(a: v2f64, b: v2f64) -> v2f64;
    // FIXME: 16-bit floats
    // #[link_name = "llvm.mips.fexdo.h"]
    // fn msa_fexdo_h(a: v4f32, b: v4f32) -> f16x8;
    #[link_name = "llvm.mips.fexdo.w"]
    fn msa_fexdo_w(a: v2f64, b: v2f64) -> v4f32;
    #[link_name = "llvm.mips.fexp2.w"]
    fn msa_fexp2_w(a: v4f32, b: v4i32) -> v4f32;
    #[link_name = "llvm.mips.fexp2.d"]
    fn msa_fexp2_d(a: v2f64, b: v2i64) -> v2f64;
    // FIXME: 16-bit floats
    // #[link_name = "llvm.mips.fexupl.w"]
    // fn msa_fexupl_w(a: f16x8) -> v4f32;
    #[link_name = "llvm.mips.fexupl.d"]
    fn msa_fexupl_d(a: v4f32) -> v2f64;
    // FIXME: 16-bit floats
    // #[link_name = "llvm.mips.fexupr.w"]
    // fn msa_fexupr_w(a: f16x8) -> v4f32;
    #[link_name = "llvm.mips.fexupr.d"]
    fn msa_fexupr_d(a: v4f32) -> v2f64;
    #[link_name = "llvm.mips.ffint.s.w"]
    fn msa_ffint_s_w(a: v4i32) -> v4f32;
    #[link_name = "llvm.mips.ffint.s.d"]
    fn msa_ffint_s_d(a: v2i64) -> v2f64;
    #[link_name = "llvm.mips.ffint.u.w"]
    fn msa_ffint_u_w(a: v4u32) -> v4f32;
    #[link_name = "llvm.mips.ffint.u.d"]
    fn msa_ffint_u_d(a: v2u64) -> v2f64;
    #[link_name = "llvm.mips.ffql.w"]
    fn msa_ffql_w(a: v8i16) -> v4f32;
    #[link_name = "llvm.mips.ffql.d"]
    fn msa_ffql_d(a: v4i32) -> v2f64;
    #[link_name = "llvm.mips.ffqr.w"]
    fn msa_ffqr_w(a: v8i16) -> v4f32;
    #[link_name = "llvm.mips.ffqr.d"]
    fn msa_ffqr_d(a: v4i32) -> v2f64;
    #[link_name = "llvm.mips.fill.b"]
    fn msa_fill_b(a: i32) -> v16i8;
    #[link_name = "llvm.mips.fill.h"]
    fn msa_fill_h(a: i32) -> v8i16;
    #[link_name = "llvm.mips.fill.w"]
    fn msa_fill_w(a: i32) -> v4i32;
    #[link_name = "llvm.mips.fill.d"]
    fn msa_fill_d(a: i64) -> v2i64;
    #[link_name = "llvm.mips.flog2.w"]
    fn msa_flog2_w(a: v4f32) -> v4f32;
    #[link_name = "llvm.mips.flog2.d"]
    fn msa_flog2_d(a: v2f64) -> v2f64;
    #[link_name = "llvm.mips.fmadd.w"]
    fn msa_fmadd_w(a: v4f32, b: v4f32, c: v4f32) -> v4f32;
    #[link_name = "llvm.mips.fmadd.d"]
    fn msa_fmadd_d(a: v2f64, b: v2f64, c: v2f64) -> v2f64;
    #[link_name = "llvm.mips.fmax.w"]
    fn msa_fmax_w(a: v4f32, b: v4f32) -> v4f32;
    #[link_name = "llvm.mips.fmax.d"]
    fn msa_fmax_d(a: v2f64, b: v2f64) -> v2f64;
    #[link_name = "llvm.mips.fmax.a.w"]
    fn msa_fmax_a_w(a: v4f32, b: v4f32) -> v4f32;
    #[link_name = "llvm.mips.fmax.a.d"]
    fn msa_fmax_a_d(a: v2f64, b: v2f64) -> v2f64;
    #[link_name = "llvm.mips.fmin.w"]
    fn msa_fmin_w(a: v4f32, b: v4f32) -> v4f32;
    #[link_name = "llvm.mips.fmin.d"]
    fn msa_fmin_d(a: v2f64, b: v2f64) -> v2f64;
    #[link_name = "llvm.mips.fmin.a.w"]
    fn msa_fmin_a_w(a: v4f32, b: v4f32) -> v4f32;
    #[link_name = "llvm.mips.fmin.a.d"]
    fn msa_fmin_a_d(a: v2f64, b: v2f64) -> v2f64;
    #[link_name = "llvm.mips.fmsub.w"]
    fn msa_fmsub_w(a: v4f32, b: v4f32, c: v4f32) -> v4f32;
    #[link_name = "llvm.mips.fmsub.d"]
    fn msa_fmsub_d(a: v2f64, b: v2f64, c: v2f64) -> v2f64;
    #[link_name = "llvm.mips.fmul.w"]
    fn msa_fmul_w(a: v4f32, b: v4f32) -> v4f32;
    #[link_name = "llvm.mips.fmul.d"]
    fn msa_fmul_d(a: v2f64, b: v2f64) -> v2f64;
    #[link_name = "llvm.mips.frint.w"]
    fn msa_frint_w(a: v4f32) -> v4f32;
    #[link_name = "llvm.mips.frint.d"]
    fn msa_frint_d(a: v2f64) -> v2f64;
    #[link_name = "llvm.mips.frcp.w"]
    fn msa_frcp_w(a: v4f32) -> v4f32;
    #[link_name = "llvm.mips.frcp.d"]
    fn msa_frcp_d(a: v2f64) -> v2f64;
    #[link_name = "llvm.mips.frsqrt.w"]
    fn msa_frsqrt_w(a: v4f32) -> v4f32;
    #[link_name = "llvm.mips.frsqrt.d"]
    fn msa_frsqrt_d(a: v2f64) -> v2f64;
    #[link_name = "llvm.mips.fsaf.w"]
    fn msa_fsaf_w(a: v4f32, b: v4f32) -> v4i32;
    #[link_name = "llvm.mips.fsaf.d"]
    fn msa_fsaf_d(a: v2f64, b: v2f64) -> v2i64;
    #[link_name = "llvm.mips.fseq.w"]
    fn msa_fseq_w(a: v4f32, b: v4f32) -> v4i32;
    #[link_name = "llvm.mips.fseq.d"]
    fn msa_fseq_d(a: v2f64, b: v2f64) -> v2i64;
    #[link_name = "llvm.mips.fsle.w"]
    fn msa_fsle_w(a: v4f32, b: v4f32) -> v4i32;
    #[link_name = "llvm.mips.fsle.d"]
    fn msa_fsle_d(a: v2f64, b: v2f64) -> v2i64;
    #[link_name = "llvm.mips.fslt.w"]
    fn msa_fslt_w(a: v4f32, b: v4f32) -> v4i32;
    #[link_name = "llvm.mips.fslt.d"]
    fn msa_fslt_d(a: v2f64, b: v2f64) -> v2i64;
    #[link_name = "llvm.mips.fsne.w"]
    fn msa_fsne_w(a: v4f32, b: v4f32) -> v4i32;
    #[link_name = "llvm.mips.fsne.d"]
    fn msa_fsne_d(a: v2f64, b: v2f64) -> v2i64;
    #[link_name = "llvm.mips.fsor.w"]
    fn msa_fsor_w(a: v4f32, b: v4f32) -> v4i32;
    #[link_name = "llvm.mips.fsor.d"]
    fn msa_fsor_d(a: v2f64, b: v2f64) -> v2i64;
    #[link_name = "llvm.mips.fsqrt.w"]
    fn msa_fsqrt_w(a: v4f32) -> v4f32;
    #[link_name = "llvm.mips.fsqrt.d"]
    fn msa_fsqrt_d(a: v2f64) -> v2f64;
    #[link_name = "llvm.mips.fsub.w"]
    fn msa_fsub_w(a: v4f32, b: v4f32) -> v4f32;
    #[link_name = "llvm.mips.fsub.d"]
    fn msa_fsub_d(a: v2f64, b: v2f64) -> v2f64;
    #[link_name = "llvm.mips.fsueq.w"]
    fn msa_fsueq_w(a: v4f32, b: v4f32) -> v4i32;
    #[link_name = "llvm.mips.fsueq.d"]
    fn msa_fsueq_d(a: v2f64, b: v2f64) -> v2i64;
    #[link_name = "llvm.mips.fsule.w"]
    fn msa_fsule_w(a: v4f32, b: v4f32) -> v4i32;
    #[link_name = "llvm.mips.fsule.d"]
    fn msa_fsule_d(a: v2f64, b: v2f64) -> v2i64;
    #[link_name = "llvm.mips.fsult.w"]
    fn msa_fsult_w(a: v4f32, b: v4f32) -> v4i32;
    #[link_name = "llvm.mips.fsult.d"]
    fn msa_fsult_d(a: v2f64, b: v2f64) -> v2i64;
    #[link_name = "llvm.mips.fsun.w"]
    fn msa_fsun_w(a: v4f32, b: v4f32) -> v4i32;
    #[link_name = "llvm.mips.fsun.d"]
    fn msa_fsun_d(a: v2f64, b: v2f64) -> v2i64;
    #[link_name = "llvm.mips.fsune.w"]
    fn msa_fsune_w(a: v4f32, b: v4f32) -> v4i32;
    #[link_name = "llvm.mips.fsune.d"]
    fn msa_fsune_d(a: v2f64, b: v2f64) -> v2i64;
    #[link_name = "llvm.mips.ftint.s.w"]
    fn msa_ftint_s_w(a: v4f32) -> v4i32;
    #[link_name = "llvm.mips.ftint.s.d"]
    fn msa_ftint_s_d(a: v2f64) -> v2i64;
    #[link_name = "llvm.mips.ftint.u.w"]
    fn msa_ftint_u_w(a: v4f32) -> v4u32;
    #[link_name = "llvm.mips.ftint.u.d"]
    fn msa_ftint_u_d(a: v2f64) -> v2u64;
    #[link_name = "llvm.mips.ftq.h"]
    fn msa_ftq_h(a: v4f32, b: v4f32) -> v8i16;
    #[link_name = "llvm.mips.ftq.w"]
    fn msa_ftq_w(a: v2f64, b: v2f64) -> v4i32;
    #[link_name = "llvm.mips.ftrunc.s.w"]
    fn msa_ftrunc_s_w(a: v4f32) -> v4i32;
    #[link_name = "llvm.mips.ftrunc.s.d"]
    fn msa_ftrunc_s_d(a: v2f64) -> v2i64;
    #[link_name = "llvm.mips.ftrunc.u.w"]
    fn msa_ftrunc_u_w(a: v4f32) -> v4u32;
    #[link_name = "llvm.mips.ftrunc.u.d"]
    fn msa_ftrunc_u_d(a: v2f64) -> v2u64;
    #[link_name = "llvm.mips.hadd.s.h"]
    fn msa_hadd_s_h(a: v16i8, b: v16i8) -> v8i16;
    #[link_name = "llvm.mips.hadd.s.w"]
    fn msa_hadd_s_w(a: v8i16, b: v8i16) -> v4i32;
    #[link_name = "llvm.mips.hadd.s.d"]
    fn msa_hadd_s_d(a: v4i32, b: v4i32) -> v2i64;
    #[link_name = "llvm.mips.hadd.u.h"]
    fn msa_hadd_u_h(a: v16u8, b: v16u8) -> v8u16;
    #[link_name = "llvm.mips.hadd.u.w"]
    fn msa_hadd_u_w(a: v8u16, b: v8u16) -> v4u32;
    #[link_name = "llvm.mips.hadd.u.d"]
    fn msa_hadd_u_d(a: v4u32, b: v4u32) -> v2u64;
    #[link_name = "llvm.mips.hsub.s.h"]
    fn msa_hsub_s_h(a: v16i8, b: v16i8) -> v8i16;
    #[link_name = "llvm.mips.hsub.s.w"]
    fn msa_hsub_s_w(a: v8i16, b: v8i16) -> v4i32;
    #[link_name = "llvm.mips.hsub.s.d"]
    fn msa_hsub_s_d(a: v4i32, b: v4i32) -> v2i64;
    #[link_name = "llvm.mips.hsub.u.h"]
    fn msa_hsub_u_h(a: v16u8, b: v16u8) -> v8i16;
    #[link_name = "llvm.mips.hsub.u.w"]
    fn msa_hsub_u_w(a: v8u16, b: v8u16) -> v4i32;
    #[link_name = "llvm.mips.hsub.u.d"]
    fn msa_hsub_u_d(a: v4u32, b: v4u32) -> v2i64;
    #[link_name = "llvm.mips.ilvev.b"]
    fn msa_ilvev_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.mips.ilvev.h"]
    fn msa_ilvev_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.mips.ilvev.w"]
    fn msa_ilvev_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.mips.ilvev.d"]
    fn msa_ilvev_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.mips.ilvl.b"]
    fn msa_ilvl_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.mips.ilvl.h"]
    fn msa_ilvl_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.mips.ilvl.w"]
    fn msa_ilvl_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.mips.ilvl.d"]
    fn msa_ilvl_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.mips.ilvod.b"]
    fn msa_ilvod_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.mips.ilvod.h"]
    fn msa_ilvod_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.mips.ilvod.w"]
    fn msa_ilvod_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.mips.ilvod.d"]
    fn msa_ilvod_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.mips.ilvr.b"]
    fn msa_ilvr_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.mips.ilvr.h"]
    fn msa_ilvr_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.mips.ilvr.w"]
    fn msa_ilvr_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.mips.ilvr.d"]
    fn msa_ilvr_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.mips.insert.b"]
    fn msa_insert_b(a: v16i8, b: i32, c: i32) -> v16i8;
    #[link_name = "llvm.mips.insert.h"]
    fn msa_insert_h(a: v8i16, b: i32, c: i32) -> v8i16;
    #[link_name = "llvm.mips.insert.w"]
    fn msa_insert_w(a: v4i32, b: i32, c: i32) -> v4i32;
    #[link_name = "llvm.mips.insert.d"]
    fn msa_insert_d(a: v2i64, b: i32, c: i64) -> v2i64;
    #[link_name = "llvm.mips.insve.b"]
    fn msa_insve_b(a: v16i8, b: i32, c: v16i8) -> v16i8;
    #[link_name = "llvm.mips.insve.h"]
    fn msa_insve_h(a: v8i16, b: i32, c: v8i16) -> v8i16;
    #[link_name = "llvm.mips.insve.w"]
    fn msa_insve_w(a: v4i32, b: i32, c: v4i32) -> v4i32;
    #[link_name = "llvm.mips.insve.d"]
    fn msa_insve_d(a: v2i64, b: i32, c: v2i64) -> v2i64;
    #[link_name = "llvm.mips.ld.b"]
    fn msa_ld_b(mem_addr: *mut u8, b: i32) -> v16i8;
    #[link_name = "llvm.mips.ld.h"]
    fn msa_ld_h(mem_addr: *mut u8, b: i32) -> v8i16;
    #[link_name = "llvm.mips.ld.w"]
    fn msa_ld_w(mem_addr: *mut u8, b: i32) -> v4i32;
    #[link_name = "llvm.mips.ld.d"]
    fn msa_ld_d(mem_addr: *mut u8, b: i32) -> v2i64;
    #[link_name = "llvm.mips.ldi.b"]
    fn msa_ldi_b(a: i32) -> v16i8;
    #[link_name = "llvm.mips.ldi.h"]
    fn msa_ldi_h(a: i32) -> v8i16;
    #[link_name = "llvm.mips.ldi.w"]
    fn msa_ldi_w(a: i32) -> v4i32;
    #[link_name = "llvm.mips.ldi.d"]
    fn msa_ldi_d(a: i32) -> v2i64;
    #[link_name = "llvm.mips.madd.q.h"]
    fn msa_madd_q_h(a: v8i16, b: v8i16, c: v8i16) -> v8i16;
    #[link_name = "llvm.mips.madd.q.w"]
    fn msa_madd_q_w(a: v4i32, b: v4i32, c: v4i32) -> v4i32;
    #[link_name = "llvm.mips.maddr.q.h"]
    fn msa_maddr_q_h(a: v8i16, b: v8i16, c: v8i16) -> v8i16;
    #[link_name = "llvm.mips.maddr.q.w"]
    fn msa_maddr_q_w(a: v4i32, b: v4i32, c: v4i32) -> v4i32;
    #[link_name = "llvm.mips.maddv.b"]
    fn msa_maddv_b(a: v16i8, b: v16i8, c: v16i8) -> v16i8;
    #[link_name = "llvm.mips.maddv.h"]
    fn msa_maddv_h(a: v8i16, b: v8i16, c: v8i16) -> v8i16;
    #[link_name = "llvm.mips.maddv.w"]
    fn msa_maddv_w(a: v4i32, b: v4i32, c: v4i32) -> v4i32;
    #[link_name = "llvm.mips.maddv.d"]
    fn msa_maddv_d(a: v2i64, b: v2i64, c: v2i64) -> v2i64;
    #[link_name = "llvm.mips.max.a.b"]
    fn msa_max_a_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.mips.max.a.h"]
    fn msa_max_a_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.mips.max.a.w"]
    fn msa_max_a_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.mips.max.a.d"]
    fn msa_max_a_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.mips.max.s.b"]
    fn msa_max_s_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.mips.max.s.h"]
    fn msa_max_s_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.mips.max.s.w"]
    fn msa_max_s_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.mips.max.s.d"]
    fn msa_max_s_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.mips.max.u.b"]
    fn msa_max_u_b(a: v16u8, b: v16u8) -> v16u8;
    #[link_name = "llvm.mips.max.u.h"]
    fn msa_max_u_h(a: v8u16, b: v8u16) -> v8u16;
    #[link_name = "llvm.mips.max.u.w"]
    fn msa_max_u_w(a: v4u32, b: v4u32) -> v4u32;
    #[link_name = "llvm.mips.max.u.d"]
    fn msa_max_u_d(a: v2u64, b: v2u64) -> v2u64;
    #[link_name = "llvm.mips.maxi.s.b"]
    fn msa_maxi_s_b(a: v16i8, b: i32) -> v16i8;
    #[link_name = "llvm.mips.maxi.s.h"]
    fn msa_maxi_s_h(a: v8i16, b: i32) -> v8i16;
    #[link_name = "llvm.mips.maxi.s.w"]
    fn msa_maxi_s_w(a: v4i32, b: i32) -> v4i32;
    #[link_name = "llvm.mips.maxi.s.d"]
    fn msa_maxi_s_d(a: v2i64, b: i32) -> v2i64;
    #[link_name = "llvm.mips.maxi.u.b"]
    fn msa_maxi_u_b(a: v16u8, b: i32) -> v16u8;
    #[link_name = "llvm.mips.maxi.u.h"]
    fn msa_maxi_u_h(a: v8u16, b: i32) -> v8u16;
    #[link_name = "llvm.mips.maxi.u.w"]
    fn msa_maxi_u_w(a: v4u32, b: i32) -> v4u32;
    #[link_name = "llvm.mips.maxi.u.d"]
    fn msa_maxi_u_d(a: v2u64, b: i32) -> v2u64;
    #[link_name = "llvm.mips.min.a.b"]
    fn msa_min_a_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.mips.min.a.h"]
    fn msa_min_a_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.mips.min.a.w"]
    fn msa_min_a_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.mips.min.a.d"]
    fn msa_min_a_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.mips.min.s.b"]
    fn msa_min_s_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.mips.min.s.h"]
    fn msa_min_s_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.mips.min.s.w"]
    fn msa_min_s_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.mips.min.s.d"]
    fn msa_min_s_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.mips.min.u.b"]
    fn msa_min_u_b(a: v16u8, b: v16u8) -> v16u8;
    #[link_name = "llvm.mips.min.u.h"]
    fn msa_min_u_h(a: v8u16, b: v8u16) -> v8u16;
    #[link_name = "llvm.mips.min.u.w"]
    fn msa_min_u_w(a: v4u32, b: v4u32) -> v4u32;
    #[link_name = "llvm.mips.min.u.d"]
    fn msa_min_u_d(a: v2u64, b: v2u64) -> v2u64;
    #[link_name = "llvm.mips.mini.s.b"]
    fn msa_mini_s_b(a: v16i8, b: i32) -> v16i8;
    #[link_name = "llvm.mips.mini.s.h"]
    fn msa_mini_s_h(a: v8i16, b: i32) -> v8i16;
    #[link_name = "llvm.mips.mini.s.w"]
    fn msa_mini_s_w(a: v4i32, b: i32) -> v4i32;
    #[link_name = "llvm.mips.mini.s.d"]
    fn msa_mini_s_d(a: v2i64, b: i32) -> v2i64;
    #[link_name = "llvm.mips.mini.u.b"]
    fn msa_mini_u_b(a: v16u8, b: i32) -> v16u8;
    #[link_name = "llvm.mips.mini.u.h"]
    fn msa_mini_u_h(a: v8u16, b: i32) -> v8u16;
    #[link_name = "llvm.mips.mini.u.w"]
    fn msa_mini_u_w(a: v4u32, b: i32) -> v4u32;
    #[link_name = "llvm.mips.mini.u.d"]
    fn msa_mini_u_d(a: v2u64, b: i32) -> v2u64;
    #[link_name = "llvm.mips.mod.s.b"]
    fn msa_mod_s_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.mips.mod.s.h"]
    fn msa_mod_s_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.mips.mod.s.w"]
    fn msa_mod_s_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.mips.mod.s.d"]
    fn msa_mod_s_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.mips.mod.u.b"]
    fn msa_mod_u_b(a: v16u8, b: v16u8) -> v16u8;
    #[link_name = "llvm.mips.mod.u.h"]
    fn msa_mod_u_h(a: v8u16, b: v8u16) -> v8u16;
    #[link_name = "llvm.mips.mod.u.w"]
    fn msa_mod_u_w(a: v4u32, b: v4u32) -> v4u32;
    #[link_name = "llvm.mips.mod.u.d"]
    fn msa_mod_u_d(a: v2u64, b: v2u64) -> v2u64;
    #[link_name = "llvm.mips.move.v"]
    fn msa_move_v(a: v16i8) -> v16i8;
    #[link_name = "llvm.mips.msub.q.h"]
    fn msa_msub_q_h(a: v8i16, b: v8i16, c: v8i16) -> v8i16;
    #[link_name = "llvm.mips.msub.q.w"]
    fn msa_msub_q_w(a: v4i32, b: v4i32, c: v4i32) -> v4i32;
    #[link_name = "llvm.mips.msubr.q.h"]
    fn msa_msubr_q_h(a: v8i16, b: v8i16, c: v8i16) -> v8i16;
    #[link_name = "llvm.mips.msubr.q.w"]
    fn msa_msubr_q_w(a: v4i32, b: v4i32, c: v4i32) -> v4i32;
    #[link_name = "llvm.mips.msubv.b"]
    fn msa_msubv_b(a: v16i8, b: v16i8, c: v16i8) -> v16i8;
    #[link_name = "llvm.mips.msubv.h"]
    fn msa_msubv_h(a: v8i16, b: v8i16, c: v8i16) -> v8i16;
    #[link_name = "llvm.mips.msubv.w"]
    fn msa_msubv_w(a: v4i32, b: v4i32, c: v4i32) -> v4i32;
    #[link_name = "llvm.mips.msubv.d"]
    fn msa_msubv_d(a: v2i64, b: v2i64, c: v2i64) -> v2i64;
    #[link_name = "llvm.mips.mul.q.h"]
    fn msa_mul_q_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.mips.mul.q.w"]
    fn msa_mul_q_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.mips.mulr.q.h"]
    fn msa_mulr_q_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.mips.mulr.q.w"]
    fn msa_mulr_q_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.mips.mulv.b"]
    fn msa_mulv_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.mips.mulv.h"]
    fn msa_mulv_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.mips.mulv.w"]
    fn msa_mulv_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.mips.mulv.d"]
    fn msa_mulv_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.mips.nloc.b"]
    fn msa_nloc_b(a: v16i8) -> v16i8;
    #[link_name = "llvm.mips.nloc.h"]
    fn msa_nloc_h(a: v8i16) -> v8i16;
    #[link_name = "llvm.mips.nloc.w"]
    fn msa_nloc_w(a: v4i32) -> v4i32;
    #[link_name = "llvm.mips.nloc.d"]
    fn msa_nloc_d(a: v2i64) -> v2i64;
    #[link_name = "llvm.mips.nlzc.b"]
    fn msa_nlzc_b(a: v16i8) -> v16i8;
    #[link_name = "llvm.mips.nlzc.h"]
    fn msa_nlzc_h(a: v8i16) -> v8i16;
    #[link_name = "llvm.mips.nlzc.w"]
    fn msa_nlzc_w(a: v4i32) -> v4i32;
    #[link_name = "llvm.mips.nlzc.d"]
    fn msa_nlzc_d(a: v2i64) -> v2i64;
    #[link_name = "llvm.mips.nor.v"]
    fn msa_nor_v(a: v16u8, b: v16u8) -> v16u8;
    #[link_name = "llvm.mips.nori.b"]
    fn msa_nori_b(a: v16u8, b: i32) -> v16u8;
    #[link_name = "llvm.mips.or.v"]
    fn msa_or_v(a: v16u8, b: v16u8) -> v16u8;
    #[link_name = "llvm.mips.ori.b"]
    fn msa_ori_b(a: v16u8, b: i32) -> v16u8;
    #[link_name = "llvm.mips.pckev.b"]
    fn msa_pckev_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.mips.pckev.h"]
    fn msa_pckev_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.mips.pckev.w"]
    fn msa_pckev_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.mips.pckev.d"]
    fn msa_pckev_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.mips.pckod.b"]
    fn msa_pckod_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.mips.pckod.h"]
    fn msa_pckod_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.mips.pckod.w"]
    fn msa_pckod_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.mips.pckod.d"]
    fn msa_pckod_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.mips.pcnt.b"]
    fn msa_pcnt_b(a: v16i8) -> v16i8;
    #[link_name = "llvm.mips.pcnt.h"]
    fn msa_pcnt_h(a: v8i16) -> v8i16;
    #[link_name = "llvm.mips.pcnt.w"]
    fn msa_pcnt_w(a: v4i32) -> v4i32;
    #[link_name = "llvm.mips.pcnt.d"]
    fn msa_pcnt_d(a: v2i64) -> v2i64;
    #[link_name = "llvm.mips.sat.s.b"]
    fn msa_sat_s_b(a: v16i8, b: i32) -> v16i8;
    #[link_name = "llvm.mips.sat.s.h"]
    fn msa_sat_s_h(a: v8i16, b: i32) -> v8i16;
    #[link_name = "llvm.mips.sat.s.w"]
    fn msa_sat_s_w(a: v4i32, b: i32) -> v4i32;
    #[link_name = "llvm.mips.sat.s.d"]
    fn msa_sat_s_d(a: v2i64, b: i32) -> v2i64;
    #[link_name = "llvm.mips.sat.u.b"]
    fn msa_sat_u_b(a: v16u8, b: i32) -> v16u8;
    #[link_name = "llvm.mips.sat.u.h"]
    fn msa_sat_u_h(a: v8u16, b: i32) -> v8u16;
    #[link_name = "llvm.mips.sat.u.w"]
    fn msa_sat_u_w(a: v4u32, b: i32) -> v4u32;
    #[link_name = "llvm.mips.sat.u.d"]
    fn msa_sat_u_d(a: v2u64, b: i32) -> v2u64;
    #[link_name = "llvm.mips.shf.b"]
    fn msa_shf_b(a: v16i8, b: i32) -> v16i8;
    #[link_name = "llvm.mips.shf.h"]
    fn msa_shf_h(a: v8i16, b: i32) -> v8i16;
    #[link_name = "llvm.mips.shf.w"]
    fn msa_shf_w(a: v4i32, b: i32) -> v4i32;
    #[link_name = "llvm.mips.sld.b"]
    fn msa_sld_b(a: v16i8, b: v16i8, c: i32) -> v16i8;
    #[link_name = "llvm.mips.sld.h"]
    fn msa_sld_h(a: v8i16, b: v8i16, c: i32) -> v8i16;
    #[link_name = "llvm.mips.sld.w"]
    fn msa_sld_w(a: v4i32, b: v4i32, c: i32) -> v4i32;
    #[link_name = "llvm.mips.sld.d"]
    fn msa_sld_d(a: v2i64, b: v2i64, c: i32) -> v2i64;
    #[link_name = "llvm.mips.sldi.b"]
    fn msa_sldi_b(a: v16i8, b: v16i8, c: i32) -> v16i8;
    #[link_name = "llvm.mips.sldi.h"]
    fn msa_sldi_h(a: v8i16, b: v8i16, c: i32) -> v8i16;
    #[link_name = "llvm.mips.sldi.w"]
    fn msa_sldi_w(a: v4i32, b: v4i32, c: i32) -> v4i32;
    #[link_name = "llvm.mips.sldi.d"]
    fn msa_sldi_d(a: v2i64, b: v2i64, c: i32) -> v2i64;
    #[link_name = "llvm.mips.sll.b"]
    fn msa_sll_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.mips.sll.h"]
    fn msa_sll_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.mips.sll.w"]
    fn msa_sll_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.mips.sll.d"]
    fn msa_sll_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.mips.slli.b"]
    fn msa_slli_b(a: v16i8, b: i32) -> v16i8;
    #[link_name = "llvm.mips.slli.h"]
    fn msa_slli_h(a: v8i16, b: i32) -> v8i16;
    #[link_name = "llvm.mips.slli.w"]
    fn msa_slli_w(a: v4i32, b: i32) -> v4i32;
    #[link_name = "llvm.mips.slli.d"]
    fn msa_slli_d(a: v2i64, b: i32) -> v2i64;
    #[link_name = "llvm.mips.splat.b"]
    fn msa_splat_b(a: v16i8, c: i32) -> v16i8;
    #[link_name = "llvm.mips.splat.h"]
    fn msa_splat_h(a: v8i16, c: i32) -> v8i16;
    #[link_name = "llvm.mips.splat.w"]
    fn msa_splat_w(a: v4i32, w: i32) -> v4i32;
    #[link_name = "llvm.mips.splat.d"]
    fn msa_splat_d(a: v2i64, c: i32) -> v2i64;
    #[link_name = "llvm.mips.splati.b"]
    fn msa_splati_b(a: v16i8, b: i32) -> v16i8;
    #[link_name = "llvm.mips.splati.h"]
    fn msa_splati_h(a: v8i16, b: i32) -> v8i16;
    #[link_name = "llvm.mips.splati.w"]
    fn msa_splati_w(a: v4i32, b: i32) -> v4i32;
    #[link_name = "llvm.mips.splati.d"]
    fn msa_splati_d(a: v2i64, b: i32) -> v2i64;
    #[link_name = "llvm.mips.sra.b"]
    fn msa_sra_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.mips.sra.h"]
    fn msa_sra_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.mips.sra.w"]
    fn msa_sra_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.mips.sra.d"]
    fn msa_sra_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.mips.srai.b"]
    fn msa_srai_b(a: v16i8, b: i32) -> v16i8;
    #[link_name = "llvm.mips.srai.h"]
    fn msa_srai_h(a: v8i16, b: i32) -> v8i16;
    #[link_name = "llvm.mips.srai.w"]
    fn msa_srai_w(a: v4i32, b: i32) -> v4i32;
    #[link_name = "llvm.mips.srai.d"]
    fn msa_srai_d(a: v2i64, b: i32) -> v2i64;
    #[link_name = "llvm.mips.srar.b"]
    fn msa_srar_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.mips.srar.h"]
    fn msa_srar_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.mips.srar.w"]
    fn msa_srar_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.mips.srar.d"]
    fn msa_srar_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.mips.srari.b"]
    fn msa_srari_b(a: v16i8, b: i32) -> v16i8;
    #[link_name = "llvm.mips.srari.h"]
    fn msa_srari_h(a: v8i16, b: i32) -> v8i16;
    #[link_name = "llvm.mips.srari.w"]
    fn msa_srari_w(a: v4i32, b: i32) -> v4i32;
    #[link_name = "llvm.mips.srari.d"]
    fn msa_srari_d(a: v2i64, b: i32) -> v2i64;
    #[link_name = "llvm.mips.srl.b"]
    fn msa_srl_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.mips.srl.h"]
    fn msa_srl_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.mips.srl.w"]
    fn msa_srl_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.mips.srl.d"]
    fn msa_srl_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.mips.srli.b"]
    fn msa_srli_b(a: v16i8, b: i32) -> v16i8;
    #[link_name = "llvm.mips.srli.h"]
    fn msa_srli_h(a: v8i16, b: i32) -> v8i16;
    #[link_name = "llvm.mips.srli.w"]
    fn msa_srli_w(a: v4i32, b: i32) -> v4i32;
    #[link_name = "llvm.mips.srli.d"]
    fn msa_srli_d(a: v2i64, b: i32) -> v2i64;
    #[link_name = "llvm.mips.srlr.b"]
    fn msa_srlr_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.mips.srlr.h"]
    fn msa_srlr_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.mips.srlr.w"]
    fn msa_srlr_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.mips.srlr.d"]
    fn msa_srlr_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.mips.srlri.b"]
    fn msa_srlri_b(a: v16i8, b: i32) -> v16i8;
    #[link_name = "llvm.mips.srlri.h"]
    fn msa_srlri_h(a: v8i16, b: i32) -> v8i16;
    #[link_name = "llvm.mips.srlri.w"]
    fn msa_srlri_w(a: v4i32, b: i32) -> v4i32;
    #[link_name = "llvm.mips.srlri.d"]
    fn msa_srlri_d(a: v2i64, b: i32) -> v2i64;
    #[link_name = "llvm.mips.st.b"]
    fn msa_st_b(a: v16i8, mem_addr: *mut u8, imm_s10: i32) -> ();
    #[link_name = "llvm.mips.st.h"]
    fn msa_st_h(a: v8i16, mem_addr: *mut u8, imm_s11: i32) -> ();
    #[link_name = "llvm.mips.st.w"]
    fn msa_st_w(a: v4i32, mem_addr: *mut u8, imm_s12: i32) -> ();
    #[link_name = "llvm.mips.st.d"]
    fn msa_st_d(a: v2i64, mem_addr: *mut u8, imm_s13: i32) -> ();
    #[link_name = "llvm.mips.subs.s.b"]
    fn msa_subs_s_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.mips.subs.s.h"]
    fn msa_subs_s_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.mips.subs.s.w"]
    fn msa_subs_s_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.mips.subs.s.d"]
    fn msa_subs_s_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.mips.subs.u.b"]
    fn msa_subs_u_b(a: v16u8, b: v16u8) -> v16u8;
    #[link_name = "llvm.mips.subs.u.h"]
    fn msa_subs_u_h(a: v8u16, b: v8u16) -> v8u16;
    #[link_name = "llvm.mips.subs.u.w"]
    fn msa_subs_u_w(a: v4u32, b: v4u32) -> v4u32;
    #[link_name = "llvm.mips.subs.u.d"]
    fn msa_subs_u_d(a: v2u64, b: v2u64) -> v2u64;
    #[link_name = "llvm.mips.subsus.u.b"]
    fn msa_subsus_u_b(a: v16u8, b: v16i8) -> v16u8;
    #[link_name = "llvm.mips.subsus.u.h"]
    fn msa_subsus_u_h(a: v8u16, b: v8i16) -> v8u16;
    #[link_name = "llvm.mips.subsus.u.w"]
    fn msa_subsus_u_w(a: v4u32, b: v4i32) -> v4u32;
    #[link_name = "llvm.mips.subsus.u.d"]
    fn msa_subsus_u_d(a: v2u64, b: v2i64) -> v2u64;
    #[link_name = "llvm.mips.subsuu.s.b"]
    fn msa_subsuu_s_b(a: v16u8, b: v16u8) -> v16i8;
    #[link_name = "llvm.mips.subsuu.s.h"]
    fn msa_subsuu_s_h(a: v8u16, b: v8u16) -> v8i16;
    #[link_name = "llvm.mips.subsuu.s.w"]
    fn msa_subsuu_s_w(a: v4u32, b: v4u32) -> v4i32;
    #[link_name = "llvm.mips.subsuu.s.d"]
    fn msa_subsuu_s_d(a: v2u64, b: v2u64) -> v2i64;
    #[link_name = "llvm.mips.subv.b"]
    fn msa_subv_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.mips.subv.h"]
    fn msa_subv_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.mips.subv.w"]
    fn msa_subv_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.mips.subv.d"]
    fn msa_subv_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.mips.subvi.b"]
    fn msa_subvi_b(a: v16i8, b: i32) -> v16i8;
    #[link_name = "llvm.mips.subvi.h"]
    fn msa_subvi_h(a: v8i16, b: i32) -> v8i16;
    #[link_name = "llvm.mips.subvi.w"]
    fn msa_subvi_w(a: v4i32, b: i32) -> v4i32;
    #[link_name = "llvm.mips.subvi.d"]
    fn msa_subvi_d(a: v2i64, b: i32) -> v2i64;
    #[link_name = "llvm.mips.vshf.b"]
    fn msa_vshf_b(a: v16i8, b: v16i8, c: v16i8) -> v16i8;
    #[link_name = "llvm.mips.vshf.h"]
    fn msa_vshf_h(a: v8i16, b: v8i16, c: v8i16) -> v8i16;
    #[link_name = "llvm.mips.vshf.w"]
    fn msa_vshf_w(a: v4i32, b: v4i32, c: v4i32) -> v4i32;
    #[link_name = "llvm.mips.vshf.d"]
    fn msa_vshf_d(a: v2i64, b: v2i64, c: v2i64) -> v2i64;
    #[link_name = "llvm.mips.xor.v"]
    fn msa_xor_v(a: v16u8, b: v16u8) -> v16u8;
    #[link_name = "llvm.mips.xori.b"]
    fn msa_xori_b(a: v16u8, b: i32) -> v16u8;
}

/// Vector Add Absolute Values.
///
/// The absolute values of the elements in vector in `a` (sixteen signed 8-bit integer numbers)
/// are added to the absolute values of the elements in vector `b` (sixteen signed 8-bit integer numbers).
/// The result is written to vector (sixteen signed 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(add_a.b))]
pub unsafe fn __msa_add_a_b(a: v16i8, b: v16i8) -> v16i8 {
    msa_add_a_b(a, mem::transmute(b))
}

/// Vector Add Absolute Values
///
/// The absolute values of the elements in vector in `a` (eight signed 16-bit integer numbers)
/// are added to the absolute values of the elements in vector `b` (eight signed 16-bit integer numbers).
/// The result is written to vector (eight signed 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(add_a.h))]
pub unsafe fn __msa_add_a_h(a: v8i16, b: v8i16) -> v8i16 {
    msa_add_a_h(a, mem::transmute(b))
}

/// Vector Add Absolute Values
///
/// The absolute values of the elements in vector in `a` (four signed 32-bit integer numbers)
/// are added to the absolute values of the elements in vector `b` (four signed 32-bit integer numbers).
/// The result is written to vector (four signed 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(add_a.w))]
pub unsafe fn __msa_add_a_w(a: v4i32, b: v4i32) -> v4i32 {
    msa_add_a_w(a, mem::transmute(b))
}

/// Vector Add Absolute Values
///
/// The absolute values of the elements in vector in `a` (two signed 64-bit integer numbers)
/// are added to the absolute values of the elements in vector `b` (two signed 64-bit integer numbers).
/// The result is written to vector (two signed 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(add_a.d))]
pub unsafe fn __msa_add_a_d(a: v2i64, b: v2i64) -> v2i64 {
    msa_add_a_d(a, mem::transmute(b))
}

/// Signed Saturated Vector Saturated Add of Absolute Values
///
/// The absolute values of the elements in vector in `a` (sixteen signed 8-bit integer numbers)
/// are added to the absolute values of the elements in vector `b` (sixteen signed 8-bit integer numbers).
/// The saturated signed result is written to vector (sixteen signed 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(adds_a.b))]
pub unsafe fn __msa_adds_a_b(a: v16i8, b: v16i8) -> v16i8 {
    msa_adds_a_b(a, mem::transmute(b))
}

/// Vector Saturated Add of Absolute Values
///
/// The absolute values of the elements in vector in `a` (eight signed 16-bit integer numbers)
/// are added to the absolute values of the elements in vector `b` (eight signed 16-bit integer numbers).
/// The saturated signed result is written to vector (eight signed 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(adds_a.h))]
pub unsafe fn __msa_adds_a_h(a: v8i16, b: v8i16) -> v8i16 {
    msa_adds_a_h(a, mem::transmute(b))
}

/// Vector Saturated Add of Absolute Values
///
/// The absolute values of the elements in vector in `a` (four signed 32-bit integer numbers)
/// are added to the absolute values of the elements in vector `b` (four signed 32-bit integer numbers).
/// The saturated signed result is written to vector (four signed 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(adds_a.w))]
pub unsafe fn __msa_adds_a_w(a: v4i32, b: v4i32) -> v4i32 {
    msa_adds_a_w(a, mem::transmute(b))
}

/// Vector Saturated Add of Absolute Values
///
/// The absolute values of the elements in vector in `a` (two signed 64-bit integer numbers)
/// are added to the absolute values of the elements in vector `b` (two signed 64-bit integer numbers).
/// The saturated signed result is written to vector (two signed 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(adds_a.d))]
pub unsafe fn __msa_adds_a_d(a: v2i64, b: v2i64) -> v2i64 {
    msa_adds_a_d(a, mem::transmute(b))
}

/// Vector Signed Saturated Add of Signed Values
///
/// The elements in vector in `a` (sixteen signed 8-bit integer numbers)
/// are added to the elements in vector `b` (sixteen signed 8-bit integer numbers).
/// Signed arithmetic is performed and overflows clamp to the largest and/or smallest
/// representable signed values before writing the result to vector (sixteen signed 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(adds_s.b))]
pub unsafe fn __msa_adds_s_b(a: v16i8, b: v16i8) -> v16i8 {
    msa_adds_s_b(a, mem::transmute(b))
}

/// Vector Signed Saturated Add of Signed Values
///
/// The elements in vector in `a` (eight signed 16-bit integer numbers)
/// are added to the elements in vector `b` (eight signed 16-bit integer numbers).
/// Signed arithmetic is performed and overflows clamp to the largest and/or smallest
/// representable signed values before writing the result to vector (eight signed 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(adds_s.h))]
pub unsafe fn __msa_adds_s_h(a: v8i16, b: v8i16) -> v8i16 {
    msa_adds_s_h(a, mem::transmute(b))
}

/// Vector Signed Saturated Add of Signed Values
///
/// The elements in vector in `a` (four signed 32-bit integer numbers)
/// are added to the elements in vector `b` (four signed 32-bit integer numbers).
/// Signed arithmetic is performed and overflows clamp to the largest and/or smallest
/// representable signed values before writing the result to vector (four signed 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(adds_s.w))]
pub unsafe fn __msa_adds_s_w(a: v4i32, b: v4i32) -> v4i32 {
    msa_adds_s_w(a, mem::transmute(b))
}

/// Vector Signed Saturated Add of Signed Values
///
/// The elements in vector in `a` (two signed 64-bit integer numbers)
/// are added to the elements in vector `b` (two signed 64-bit integer numbers).
/// Signed arithmetic is performed and overflows clamp to the largest and/or smallest
/// representable signed values before writing the result to vector (two signed 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(adds_s.d))]
pub unsafe fn __msa_adds_s_d(a: v2i64, b: v2i64) -> v2i64 {
    msa_adds_s_d(a, mem::transmute(b))
}

/// Vector Unsigned Saturated Add of Unsigned Values
///
/// The elements in vector in `a` (sixteen unsigned 8-bit integer numbers)
/// are added to the elements in vector `b` (sixteen unsigned 8-bit integer numbers).
/// Signed arithmetic is performed and overflows clamp to the largest and/or smallest
/// representable signed values before writing the result to vector (sixteen unsigned 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(adds_u.b))]
pub unsafe fn __msa_adds_u_b(a: v16u8, b: v16u8) -> v16u8 {
    msa_adds_u_b(a, mem::transmute(b))
}

/// Vector Unsigned Saturated Add of Unsigned Values
///
/// The elements in vector in `a` (eight unsigned 16-bit integer numbers)
/// are added to the elements in vector `b` (eight unsigned 16-bit integer numbers).
/// Signed arithmetic is performed and overflows clamp to the largest and/or smallest
/// representable signed values before writing the result to vector (eight unsigned 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(adds_u.h))]
pub unsafe fn __msa_adds_u_h(a: v8u16, b: v8u16) -> v8u16 {
    msa_adds_u_h(a, mem::transmute(b))
}

/// Vector Unsigned Saturated Add of Unsigned Values
///
/// The elements in vector in `a` (four unsigned 32-bit integer numbers)
/// are added to the elements in vector `b` (four unsigned 32-bit integer numbers).
/// Signed arithmetic is performed and overflows clamp to the largest and/or smallest
/// representable signed values before writing the result to vector (four unsigned 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(adds_u.w))]
pub unsafe fn __msa_adds_u_w(a: v4u32, b: v4u32) -> v4u32 {
    msa_adds_u_w(a, mem::transmute(b))
}

/// Vector Unsigned Saturated Add of Unsigned Values
///
/// The elements in vector in `a` (two unsigned 64-bit integer numbers)
/// are added to the elements in vector `b` (two unsigned 64-bit integer numbers).
/// Signed arithmetic is performed and overflows clamp to the largest and/or smallest
/// representable signed values before writing the result to vector (two unsigned 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(adds_u.d))]
pub unsafe fn __msa_adds_u_d(a: v2u64, b: v2u64) -> v2u64 {
    msa_adds_u_d(a, mem::transmute(b))
}

/// Vector Add
///
/// The elements in vector in `a` (sixteen signed 8-bit integer numbers)
/// are added to the elements in vector `b` (sixteen signed 8-bit integer numbers).
/// The result is written to vector (sixteen signed 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(addv.b))]
pub unsafe fn __msa_addv_b(a: v16i8, b: v16i8) -> v16i8 {
    msa_addv_b(a, mem::transmute(b))
}

/// Vector Add
///
/// The elements in vector in `a` (eight signed 16-bit integer numbers)
/// are added to the elements in vector `b` (eight signed 16-bit integer numbers).
/// The result is written to vector (eight signed 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(addv.h))]
pub unsafe fn __msa_addv_h(a: v8i16, b: v8i16) -> v8i16 {
    msa_addv_h(a, mem::transmute(b))
}

/// Vector Add
///
/// The elements in vector in `a` (four signed 32-bit integer numbers)
/// are added to the elements in vector `b` (four signed 32-bit integer numbers).
/// The result is written to vector (four signed 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(addv.w))]
pub unsafe fn __msa_addv_w(a: v4i32, b: v4i32) -> v4i32 {
    msa_addv_w(a, mem::transmute(b))
}

/// Vector Add
///
/// The elements in vector in `a` (two signed 64-bit integer numbers)
/// are added to the elements in vector `b` (two signed 64-bit integer numbers).
/// The result is written to vector (two signed 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(addv.d))]
pub unsafe fn __msa_addv_d(a: v2i64, b: v2i64) -> v2i64 {
    msa_addv_d(a, mem::transmute(b))
}

/// Immediate Add
///
/// The 5-bit immediate unsigned value `imm5` is added to the elements
/// vector in `a` (sixteen signed 8-bit integer numbers).
/// The result is written to vector (sixteen signed 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(addvi.b, imm5 = 0b10111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_addvi_b<const IMM5: i32>(a: v16i8) -> v16i8 {
    static_assert_imm5!(IMM5);
    msa_addvi_b(a, IMM5)
}

/// Immediate Add
///
/// The 5-bit immediate unsigned value `imm5` is added to the elements
/// vector in `a` (eight signed 16-bit integer numbers).
/// The result is written to vector (eight signed 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(addvi.h, imm5 = 0b10111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_addvi_h<const IMM5: i32>(a: v8i16) -> v8i16 {
    static_assert_imm5!(IMM5);
    msa_addvi_h(a, IMM5)
}

/// Immediate Add
///
/// The 5-bit immediate unsigned value `imm5` is added to the elements
/// vector in `a` (four signed 32-bit integer numbers).
/// The result is written to vector (four signed 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(addvi.w, imm5 = 0b10111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_addvi_w<const IMM5: i32>(a: v4i32) -> v4i32 {
    static_assert_imm5!(IMM5);
    msa_addvi_w(a, IMM5)
}

/// Immediate Add
///
/// The 5-bit immediate unsigned value `imm5` is added to the elements
/// vector in `a` (two signed 64-bit integer numbers).
/// The result is written to vector (two signed 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(addvi.d, imm5 = 0b10111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_addvi_d<const IMM5: i32>(a: v2i64) -> v2i64 {
    static_assert_imm5!(IMM5);
    msa_addvi_d(a, IMM5)
}

/// Vector Logical And
///
/// Each bit of vector `a` (sixteen unsigned 8-bit integer numbers)
/// is combined with the corresponding bit of vector `b` (sixteen unsigned 8-bit integer numbers)
/// in a bitwise logical AND operation.
/// The result is written to vector (sixteen unsigned 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(and.v))]
pub unsafe fn __msa_and_v(a: v16u8, b: v16u8) -> v16u8 {
    msa_and_v(a, mem::transmute(b))
}

/// Immediate Logical And
///
/// Each byte element of vector `a` (sixteen unsigned 8-bit integer numbers)
/// is combined with the 8-bit immediate i8 (signed 8-bit integer number) in a bitwise logical AND operation.
/// The result is written to vector (sixteen unsigned 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(andi.b, imm8 = 0b10010111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_andi_b<const IMM8: i32>(a: v16u8) -> v16u8 {
    static_assert_imm8!(IMM8);
    msa_andi_b(a, IMM8)
}

/// Vector Absolute Values of Signed Subtract
///
/// The signed elements in vector `a` (sixteen signed 8-bit integer numbers)
/// are subtracted from the signed elements in vector `b` (sixteen signed 8-bit integer numbers).
/// The absolute value of the signed result is written to vector (sixteen signed 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(asub_s.b))]
pub unsafe fn __msa_asub_s_b(a: v16i8, b: v16i8) -> v16i8 {
    msa_asub_s_b(a, mem::transmute(b))
}

/// Vector Absolute Values of Signed Subtract
///
/// The signed elements in vector `a` (eight signed 16-bit integer numbers)
/// are subtracted from the signed elements in vector `b` (eight signed 16-bit integer numbers).
/// The absolute value of the signed result is written to vector (eight signed 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(asub_s.h))]
pub unsafe fn __msa_asub_s_h(a: v8i16, b: v8i16) -> v8i16 {
    msa_asub_s_h(a, mem::transmute(b))
}

/// Vector Absolute Values of Signed Subtract
///
/// The signed elements in vector `a` (four signed 32-bit integer numbers)
/// are subtracted from the signed elements in vector `b` (four signed 32-bit integer numbers).
/// The absolute value of the signed result is written to vector (four signed 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(asub_s.w))]
pub unsafe fn __msa_asub_s_w(a: v4i32, b: v4i32) -> v4i32 {
    msa_asub_s_w(a, mem::transmute(b))
}

/// Vector Absolute Values of Signed Subtract
///
/// The signed elements in vector `a` (two signed 64-bit integer numbers)
/// are subtracted from the signed elements in vector `b` (two signed 64-bit integer numbers).
/// The absolute value of the signed result is written to vector (two signed 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(asub_s.d))]
pub unsafe fn __msa_asub_s_d(a: v2i64, b: v2i64) -> v2i64 {
    msa_asub_s_d(a, mem::transmute(b))
}

/// Vector Absolute Values of Unsigned Subtract
///
/// The unsigned elements in vector `a` (sixteen unsigned 8-bit integer numbers)
/// are subtracted from the unsigned elements in vector `b` (sixteen unsigned 8-bit integer numbers).
/// The absolute value of the unsigned result is written to vector (sixteen unsigned 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(asub_u.b))]
pub unsafe fn __msa_asub_u_b(a: v16u8, b: v16u8) -> v16u8 {
    msa_asub_u_b(a, mem::transmute(b))
}

/// Vector Absolute Values of Unsigned Subtract
///
/// The unsigned elements in vector `a` (eight unsigned 16-bit integer numbers)
/// are subtracted from the unsigned elements in vector `b` (eight unsigned 16-bit integer numbers).
/// The absolute value of the unsigned result is written to vector (eight unsigned 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(asub_u.h))]
pub unsafe fn __msa_asub_u_h(a: v8u16, b: v8u16) -> v8u16 {
    msa_asub_u_h(a, mem::transmute(b))
}

/// Vector Absolute Values of Unsigned Subtract
///
/// The unsigned elements in vector `a` (four unsigned 32-bit integer numbers)
/// are subtracted from the unsigned elements in vector `b` (four unsigned 32-bit integer numbers).
/// The absolute value of the unsigned result is written to vector (four unsigned 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(asub_u.w))]
pub unsafe fn __msa_asub_u_w(a: v4u32, b: v4u32) -> v4u32 {
    msa_asub_u_w(a, mem::transmute(b))
}

/// Vector Absolute Values of Unsigned Subtract
///
/// The unsigned elements in vector `a` (two unsigned 64-bit integer numbers)
/// are subtracted from the unsigned elements in vector `b` (two unsigned 64-bit integer numbers).
/// The absolute value of the unsigned result is written to vector (two unsigned 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(asub_u.d))]
pub unsafe fn __msa_asub_u_d(a: v2u64, b: v2u64) -> v2u64 {
    msa_asub_u_d(a, mem::transmute(b))
}

/// Vector Signed Average
///
/// The elements in vector `a` (sixteen signed 8-bit integer numbers)
/// are added to the elements in vector `b` (sixteen signed 8-bit integer numbers).
/// The addition is done signed with full precision, i.e. the result has one extra bit.
/// Signed division by 2 (or arithmetic shift right by one bit) is performed before
/// writing the result to vector (sixteen signed 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ave_s.b))]
pub unsafe fn __msa_ave_s_b(a: v16i8, b: v16i8) -> v16i8 {
    msa_ave_s_b(a, mem::transmute(b))
}

/// Vector Signed Average
///
/// The elements in vector `a` (eight signed 16-bit integer numbers)
/// are added to the elements in vector `b` (eight signed 16-bit integer numbers).
/// The addition is done signed with full precision, i.e. the result has one extra bit.
/// Signed division by 2 (or arithmetic shift right by one bit) is performed before
/// writing the result to vector (eight signed 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ave_s.h))]
pub unsafe fn __msa_ave_s_h(a: v8i16, b: v8i16) -> v8i16 {
    msa_ave_s_h(a, mem::transmute(b))
}

/// Vector Signed Average
///
/// The elements in vector `a` (four signed 32-bit integer numbers)
/// are added to the elements in vector `b` (four signed 32-bit integer numbers).
/// The addition is done signed with full precision, i.e. the result has one extra bit.
/// Signed division by 2 (or arithmetic shift right by one bit) is performed before
/// writing the result to vector (four signed 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ave_s.w))]
pub unsafe fn __msa_ave_s_w(a: v4i32, b: v4i32) -> v4i32 {
    msa_ave_s_w(a, mem::transmute(b))
}

/// Vector Signed Average
///
/// The elements in vector `a` (two signed 64-bit integer numbers)
/// are added to the elements in vector `b` (two signed 64-bit integer numbers).
/// The addition is done signed with full precision, i.e. the result has one extra bit.
/// Signed division by 2 (or arithmetic shift right by one bit) is performed before
/// writing the result to vector (two signed 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ave_s.d))]
pub unsafe fn __msa_ave_s_d(a: v2i64, b: v2i64) -> v2i64 {
    msa_ave_s_d(a, mem::transmute(b))
}

/// Vector Unsigned Average
///
/// The elements in vector `a` (sixteen unsigned 8-bit integer numbers)
/// are added to the elements in vector `b` (sixteen unsigned 8-bit integer numbers).
/// The addition is done unsigned with full precision, i.e. the result has one extra bit.
/// Unsigned division by 2 (or logical shift right by one bit) is performed before
/// writing the result to vector (sixteen unsigned 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ave_u.b))]
pub unsafe fn __msa_ave_u_b(a: v16u8, b: v16u8) -> v16u8 {
    msa_ave_u_b(a, mem::transmute(b))
}

/// Vector Unsigned Average
///
/// The elements in vector `a` (eight unsigned 16-bit integer numbers)
/// are added to the elements in vector `b` (eight unsigned 16-bit integer numbers).
/// The addition is done unsigned with full precision, i.e. the result has one extra bit.
/// Unsigned division by 2 (or logical shift right by one bit) is performed before
/// writing the result to vector (eight unsigned 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ave_u.h))]
pub unsafe fn __msa_ave_u_h(a: v8u16, b: v8u16) -> v8u16 {
    msa_ave_u_h(a, mem::transmute(b))
}

/// Vector Unsigned Average
///
/// The elements in vector `a` (four unsigned 32-bit integer numbers)
/// are added to the elements in vector `b` (four unsigned 32-bit integer numbers).
/// The addition is done unsigned with full precision, i.e. the result has one extra bit.
/// Unsigned division by 2 (or logical shift right by one bit) is performed before
/// writing the result to vector (four unsigned 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ave_u.w))]
pub unsafe fn __msa_ave_u_w(a: v4u32, b: v4u32) -> v4u32 {
    msa_ave_u_w(a, mem::transmute(b))
}

/// Vector Unsigned Average
///
/// The elements in vector `a` (two unsigned 64-bit integer numbers)
/// are added to the elements in vector `b` (two unsigned 64-bit integer numbers).
/// The addition is done unsigned with full precision, i.e. the result has one extra bit.
/// Unsigned division by 2 (or logical shift right by one bit) is performed before
/// writing the result to vector (two unsigned 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ave_u.d))]
pub unsafe fn __msa_ave_u_d(a: v2u64, b: v2u64) -> v2u64 {
    msa_ave_u_d(a, mem::transmute(b))
}

/// Vector Signed Average Rounded
///
/// The elements in vector `a` (sixteen signed 8-bit integer numbers)
/// are added to the elements in vector `b` (sixteen signed 8-bit integer numbers).
/// The addition of the elements plus 1 (for rounding) is done signed with full precision,
/// i.e. the result has one extra bit.
/// Signed division by 2 (or arithmetic shift right by one bit) is performed before
/// writing the result to vector (sixteen signed 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(aver_s.b))]
pub unsafe fn __msa_aver_s_b(a: v16i8, b: v16i8) -> v16i8 {
    msa_aver_s_b(a, mem::transmute(b))
}

/// Vector Signed Average Rounded
///
/// The elements in vector `a` (eight signed 16-bit integer numbers)
/// are added to the elements in vector `b` (eight signed 16-bit integer numbers).
/// The addition of the elements plus 1 (for rounding) is done signed with full precision,
/// i.e. the result has one extra bit.
/// Signed division by 2 (or arithmetic shift right by one bit) is performed before
/// writing the result to vector (eight signed 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(aver_s.h))]
pub unsafe fn __msa_aver_s_h(a: v8i16, b: v8i16) -> v8i16 {
    msa_aver_s_h(a, mem::transmute(b))
}

/// Vector Signed Average Rounded
///
/// The elements in vector `a` (four signed 32-bit integer numbers)
/// are added to the elements in vector `b` (four signed 32-bit integer numbers).
/// The addition of the elements plus 1 (for rounding) is done signed with full precision,
/// i.e. the result has one extra bit.
/// Signed division by 2 (or arithmetic shift right by one bit) is performed before
/// writing the result to vector (four signed 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(aver_s.w))]
pub unsafe fn __msa_aver_s_w(a: v4i32, b: v4i32) -> v4i32 {
    msa_aver_s_w(a, mem::transmute(b))
}

/// Vector Signed Average Rounded
///
/// The elements in vector `a` (two signed 64-bit integer numbers)
/// are added to the elements in vector `b` (two signed 64-bit integer numbers).
/// The addition of the elements plus 1 (for rounding) is done signed with full precision,
/// i.e. the result has one extra bit.
/// Signed division by 2 (or arithmetic shift right by one bit) is performed before
/// writing the result to vector (two signed 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(aver_s.d))]
pub unsafe fn __msa_aver_s_d(a: v2i64, b: v2i64) -> v2i64 {
    msa_aver_s_d(a, mem::transmute(b))
}

/// Vector Unsigned Average Rounded
///
/// The elements in vector `a` (sixteen unsigned 8-bit integer numbers)
/// are added to the elements in vector `b` (sixteen unsigned 8-bit integer numbers).
/// The addition of the elements plus 1 (for rounding) is done unsigned with full precision,
/// i.e. the result has one extra bit.
/// Unsigned division by 2 (or logical shift right by one bit) is performed before
/// writing the result to vector (sixteen unsigned 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(aver_u.b))]
pub unsafe fn __msa_aver_u_b(a: v16u8, b: v16u8) -> v16u8 {
    msa_aver_u_b(a, mem::transmute(b))
}

/// Vector Unsigned Average Rounded
///
/// The elements in vector `a` (eight unsigned 16-bit integer numbers)
/// are added to the elements in vector `b` (eight unsigned 16-bit integer numbers).
/// The addition of the elements plus 1 (for rounding) is done unsigned with full precision,
/// i.e. the result has one extra bit.
/// Unsigned division by 2 (or logical shift right by one bit) is performed before
/// writing the result to vector (eight unsigned 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(aver_u.h))]
pub unsafe fn __msa_aver_u_h(a: v8u16, b: v8u16) -> v8u16 {
    msa_aver_u_h(a, mem::transmute(b))
}

/// Vector Unsigned Average Rounded
///
/// The elements in vector `a` (four unsigned 32-bit integer numbers)
/// are added to the elements in vector `b` (four unsigned 32-bit integer numbers).
/// The addition of the elements plus 1 (for rounding) is done unsigned with full precision,
/// i.e. the result has one extra bit.
/// Unsigned division by 2 (or logical shift right by one bit) is performed before
/// writing the result to vector (four unsigned 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(aver_u.w))]
pub unsafe fn __msa_aver_u_w(a: v4u32, b: v4u32) -> v4u32 {
    msa_aver_u_w(a, mem::transmute(b))
}

/// Vector Unsigned Average Rounded
///
/// The elements in vector `a` (two unsigned 64-bit integer numbers)
/// are added to the elements in vector `b` (two unsigned 64-bit integer numbers).
/// The addition of the elements plus 1 (for rounding) is done unsigned with full precision,
/// i.e. the result has one extra bit.
/// Unsigned division by 2 (or logical shift right by one bit) is performed before
/// writing the result to vector (two unsigned 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(aver_u.d))]
pub unsafe fn __msa_aver_u_d(a: v2u64, b: v2u64) -> v2u64 {
    msa_aver_u_d(a, mem::transmute(b))
}

/// Vector Bit Clear
///
/// Clear (set to 0) one bit in each element of vector `a` (sixteen unsigned 8-bit integer numbers).
/// The bit position is given by the elements in `b` (sixteen unsigned 8-bit integer numbers)
/// modulo the size of the element in bits.
/// The result is written to vector (sixteen unsigned 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(bclr.b))]
pub unsafe fn __msa_bclr_b(a: v16u8, b: v16u8) -> v16u8 {
    msa_bclr_b(a, mem::transmute(b))
}

/// Vector Bit Clear
///
/// Clear (set to 0) one bit in each element of vector `a` (eight unsigned 16-bit integer numbers).
/// The bit position is given by the elements in `b` (eight unsigned 16-bit integer numbers)
/// modulo the size of the element in bits.
/// The result is written to vector (eight unsigned 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(bclr.h))]
pub unsafe fn __msa_bclr_h(a: v8u16, b: v8u16) -> v8u16 {
    msa_bclr_h(a, mem::transmute(b))
}

/// Vector Bit Clear
///
/// Clear (set to 0) one bit in each element of vector `a` (four unsigned 32-bit integer numbers).
/// The bit position is given by the elements in `b` (four unsigned 32-bit integer numbers)
/// modulo the size of the element in bits.
/// The result is written to vector (four unsigned 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(bclr.w))]
pub unsafe fn __msa_bclr_w(a: v4u32, b: v4u32) -> v4u32 {
    msa_bclr_w(a, mem::transmute(b))
}

/// Vector Bit Clear
///
/// Clear (set to 0) one bit in each element of vector `a` (two unsigned 64-bit integer numbers).
/// The bit position is given by the elements in `b` (two unsigned 64-bit integer numbers)
/// modulo the size of the element in bits.
/// The result is written to vector (two unsigned 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(bclr.d))]
pub unsafe fn __msa_bclr_d(a: v2u64, b: v2u64) -> v2u64 {
    msa_bclr_d(a, mem::transmute(b))
}

/// Immediate Bit Clear
///
/// Clear (set to 0) one bit in each element of vector `a` (sixteen unsigned 8-bit integer numbers).
/// The bit position is given by the immediate `m` modulo the size of the element in bits.
/// The result is written to vector (sixteen unsigned 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(bclri.b, imm3 = 0b111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_bclri_b<const IMM3: i32>(a: v16u8) -> v16u8 {
    static_assert_imm3!(IMM3);
    msa_bclri_b(a, IMM3)
}

/// Immediate Bit Clear
///
/// Clear (set to 0) one bit in each element of vector `a` (eight unsigned 16-bit integer numbers).
/// The bit position is given by the immediate `m` modulo the size of the element in bits.
/// The result is written to vector (eight unsigned 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(bclri.h, imm4 = 0b1111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_bclri_h<const IMM4: i32>(a: v8u16) -> v8u16 {
    static_assert_imm4!(IMM4);
    msa_bclri_h(a, IMM4)
}

/// Immediate Bit Clear
///
/// Clear (set to 0) one bit in each element of vector `a` (four unsigned 32-bit integer numbers).
/// The bit position is given by the immediate `m` modulo the size of the element in bits.
/// The result is written to vector (four unsigned 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(bclri.w, imm5 = 0b11111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_bclri_w<const IMM5: i32>(a: v4u32) -> v4u32 {
    static_assert_imm5!(IMM5);
    msa_bclri_w(a, IMM5)
}

/// Immediate Bit Clear
///
/// Clear (set to 0) one bit in each element of vector `a` (two unsigned 64-bit integer numbers).
/// The bit position is given by the immediate `m` modulo the size of the element in bits.
/// The result is written to vector (two unsigned 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(bclri.d, imm6 = 0b111111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_bclri_d<const IMM6: i32>(a: v2u64) -> v2u64 {
    static_assert_imm6!(IMM6);
    msa_bclri_d(a, IMM6)
}

/// Vector Bit Insert Left
///
/// Copy most significant (left) bits in each element of vector `b` (sixteen unsigned 8-bit integer numbers)
/// to elements in vector `a` (sixteen unsigned 8-bit integer numbers) while preserving the least significant (right) bits.
/// The number of bits to copy is given by the elements in vector `c` (sixteen unsigned 8-bit integer numbers)
/// modulo the size of the element in bits plus 1.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(binsl.b))]
pub unsafe fn __msa_binsl_b(a: v16u8, b: v16u8, c: v16u8) -> v16u8 {
    msa_binsl_b(a, mem::transmute(b), c)
}

/// Vector Bit Insert Left
///
/// Copy most significant (left) bits in each element of vector `b` (eight unsigned 16-bit integer numbers)
/// to elements in vector `a` (eight unsigned 16-bit integer numbers) while preserving the least significant (right) bits.
/// The number of bits to copy is given by the elements in vector `c` (eight unsigned 16-bit integer numbers)
/// modulo the size of the element in bits plus 1.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(binsl.h))]
pub unsafe fn __msa_binsl_h(a: v8u16, b: v8u16, c: v8u16) -> v8u16 {
    msa_binsl_h(a, mem::transmute(b), c)
}

/// Vector Bit Insert Left
///
/// Copy most significant (left) bits in each element of vector `b` (four unsigned 32-bit integer numbers)
/// to elements in vector `a` (four unsigned 32-bit integer numbers) while preserving the least significant (right) bits.
/// The number of bits to copy is given by the elements in vector `c` (four unsigned 32-bit integer numbers)
/// modulo the size of the element in bits plus 1.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(binsl.w))]
pub unsafe fn __msa_binsl_w(a: v4u32, b: v4u32, c: v4u32) -> v4u32 {
    msa_binsl_w(a, mem::transmute(b), c)
}

/// Vector Bit Insert Left
///
/// Copy most significant (left) bits in each element of vector `b` (two unsigned 64-bit integer numbers)
/// to elements in vector `a` (two unsigned 64-bit integer numbers) while preserving the least significant (right) bits.
/// The number of bits to copy is given by the elements in vector `c` (two unsigned 64-bit integer numbers)
/// modulo the size of the element in bits plus 1.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(binsl.d))]
pub unsafe fn __msa_binsl_d(a: v2u64, b: v2u64, c: v2u64) -> v2u64 {
    msa_binsl_d(a, mem::transmute(b), c)
}

/// Immediate Bit Insert Left
///
/// Copy most significant (left) bits in each element of vector `b` (sixteen unsigned 8-bit integer numbers)
/// to elements in vector `a` (sixteen unsigned 8-bit integer numbers) while preserving the least significant (right) bits.
/// The number of bits to copy is given by the immediate `imm3` modulo the size of the element in bits plus 1.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(binsli.b, imm3 = 0b111))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn __msa_binsli_b<const IMM3: i32>(a: v16u8, b: v16u8) -> v16u8 {
    static_assert_imm3!(IMM3);
    msa_binsli_b(a, mem::transmute(b), IMM3)
}

/// Immediate Bit Insert Left
///
/// Copy most significant (left) bits in each element of vector `b` (eight unsigned 16-bit integer numbers)
/// to elements in vector `a` (eight unsigned 16-bit integer numbers) while preserving the least significant (right) bits.
/// The number of bits to copy is given by the immediate `imm4` modulo the size of the element in bits plus 1.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(binsli.h, imm4 = 0b1111))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn __msa_binsli_h<const IMM4: i32>(a: v8u16, b: v8u16) -> v8u16 {
    static_assert_imm4!(IMM4);
    msa_binsli_h(a, mem::transmute(b), IMM4)
}

/// Immediate Bit Insert Left
///
/// Copy most significant (left) bits in each element of vector `b` (four unsigned 32-bit integer numbers)
/// to elements in vector `a` (four unsigned 32-bit integer numbers) while preserving the least significant (right) bits.
/// The number of bits to copy is given by the immediate `imm5` modulo the size of the element in bits plus 1.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(binsli.w, imm5 = 0b11111))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn __msa_binsli_w<const IMM5: i32>(a: v4u32, b: v4u32) -> v4u32 {
    static_assert_imm5!(IMM5);
    msa_binsli_w(a, mem::transmute(b), IMM5)
}

/// Immediate Bit Insert Left
///
/// Copy most significant (left) bits in each element of vector `b` (two unsigned 64-bit integer numbers)
/// to elements in vector `a` (two unsigned 64-bit integer numbers) while preserving the least significant (right) bits.
/// The number of bits to copy is given by the immediate `imm6` modulo the size of the element in bits plus 1.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(binsli.d, imm6 = 0b111111))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn __msa_binsli_d<const IMM6: i32>(a: v2u64, b: v2u64) -> v2u64 {
    static_assert_imm6!(IMM6);
    msa_binsli_d(a, mem::transmute(b), IMM6)
}

/// Vector Bit Insert Right
///
/// Copy most significant (right) bits in each element of vector `b` (sixteen unsigned 8-bit integer numbers)
/// to elements in vector `a` (sixteen unsigned 8-bit integer numbers) while preserving the least significant (left) bits.
/// The number of bits to copy is given by the elements in vector `c` (sixteen unsigned 8-bit integer numbers)
/// modulo the size of the element in bits plus 1.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(binsr.b))]
pub unsafe fn __msa_binsr_b(a: v16u8, b: v16u8, c: v16u8) -> v16u8 {
    msa_binsr_b(a, mem::transmute(b), c)
}

/// Vector Bit Insert Right
///
/// Copy most significant (right) bits in each element of vector `b` (eight unsigned 16-bit integer numbers)
/// to elements in vector `a` (eight unsigned 16-bit integer numbers) while preserving the least significant (left) bits.
/// The number of bits to copy is given by the elements in vector `c` (eight unsigned 16-bit integer numbers)
/// modulo the size of the element in bits plus 1.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(binsr.h))]
pub unsafe fn __msa_binsr_h(a: v8u16, b: v8u16, c: v8u16) -> v8u16 {
    msa_binsr_h(a, mem::transmute(b), c)
}

/// Vector Bit Insert Right
///
/// Copy most significant (right) bits in each element of vector `b` (four unsigned 32-bit integer numbers)
/// to elements in vector `a` (four unsigned 32-bit integer numbers) while preserving the least significant (left) bits.
/// The number of bits to copy is given by the elements in vector `c` (four unsigned 32-bit integer numbers)
/// modulo the size of the element in bits plus 1.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(binsr.w))]
pub unsafe fn __msa_binsr_w(a: v4u32, b: v4u32, c: v4u32) -> v4u32 {
    msa_binsr_w(a, mem::transmute(b), c)
}

/// Vector Bit Insert Right
///
/// Copy most significant (right) bits in each element of vector `b` (two unsigned 64-bit integer numbers)
/// to elements in vector `a` (two unsigned 64-bit integer numbers) while preserving the least significant (left) bits.
/// The number of bits to copy is given by the elements in vector `c` (two unsigned 64-bit integer numbers)
/// modulo the size of the element in bits plus 1.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(binsr.d))]
pub unsafe fn __msa_binsr_d(a: v2u64, b: v2u64, c: v2u64) -> v2u64 {
    msa_binsr_d(a, mem::transmute(b), c)
}

/// Immediate Bit Insert Right
///
/// Copy most significant (right) bits in each element of vector `b` (sixteen unsigned 8-bit integer numbers)
/// to elements in vector `a` (sixteen unsigned 8-bit integer numbers) while preserving the least significant (left) bits.
/// The number of bits to copy is given by the immediate `imm3` modulo the size of the element in bits plus 1.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(binsri.b, imm3 = 0b111))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn __msa_binsri_b<const IMM3: i32>(a: v16u8, b: v16u8) -> v16u8 {
    static_assert_imm3!(IMM3);
    msa_binsri_b(a, mem::transmute(b), IMM3)
}

/// Immediate Bit Insert Right
///
/// Copy most significant (right) bits in each element of vector `b` (eight unsigned 16-bit integer numbers)
/// to elements in vector `a` (eight unsigned 16-bit integer numbers) while preserving the least significant (left) bits.
/// The number of bits to copy is given by the immediate `imm4` modulo the size of the element in bits plus 1.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(binsri.h, imm4 = 0b1111))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn __msa_binsri_h<const IMM4: i32>(a: v8u16, b: v8u16) -> v8u16 {
    static_assert_imm4!(IMM4);
    msa_binsri_h(a, mem::transmute(b), IMM4)
}

/// Immediate Bit Insert Right
///
/// Copy most significant (right) bits in each element of vector `b` (four unsigned 32-bit integer numbers)
/// to elements in vector `a` (four unsigned 32-bit integer numbers) while preserving the least significant (left) bits.
/// The number of bits to copy is given by the immediate `imm5` modulo the size of the element in bits plus 1.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(binsri.w, imm5 = 0b11111))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn __msa_binsri_w<const IMM5: i32>(a: v4u32, b: v4u32) -> v4u32 {
    static_assert_imm5!(IMM5);
    msa_binsri_w(a, mem::transmute(b), IMM5)
}

/// Immediate Bit Insert Right
///
/// Copy most significant (right) bits in each element of vector `b` (two unsigned 64-bit integer numbers)
/// to elements in vector `a` (two unsigned 64-bit integer numbers) while preserving the least significant (left) bits.
/// The number of bits to copy is given by the immediate `imm6` modulo the size of the element in bits plus 1.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(binsri.d, imm6 = 0b111111))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn __msa_binsri_d<const IMM6: i32>(a: v2u64, b: v2u64) -> v2u64 {
    static_assert_imm6!(IMM6);
    msa_binsri_d(a, mem::transmute(b), IMM6)
}

/// Vector Bit Move If Not Zero
///
/// Copy to destination vector `a` (sixteen unsigned 8-bit integer numbers) all bits from source vector
/// `b` (sixteen unsigned 8-bit integer numbers) for which the corresponding bits from target vector `c`
/// (sixteen unsigned 8-bit integer numbers) are 1 and leaves unchanged all destination bits
/// for which the corresponding target bits are 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(bmnz.v))]
pub unsafe fn __msa_bmnz_v(a: v16u8, b: v16u8, c: v16u8) -> v16u8 {
    msa_bmnz_v(a, mem::transmute(b), c)
}

/// Immediate Bit Move If Not Zero
///
/// Copy to destination vector `a` (sixteen unsigned 8-bit integer numbers) all bits from source vector
/// `b` (sixteen unsigned 8-bit integer numbers) for which the corresponding bits from from immediate `imm8`
/// are 1 and leaves unchanged all destination bits for which the corresponding target bits are 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(bmnzi.b, imm8 = 0b11111111))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn __msa_bmnzi_b<const IMM8: i32>(a: v16u8, b: v16u8) -> v16u8 {
    static_assert_imm8!(IMM8);
    msa_bmnzi_b(a, mem::transmute(b), IMM8)
}

/// Vector Bit Move If Zero
///
/// Copy to destination vector `a` (sixteen unsigned 8-bit integer numbers) all bits from source vector
/// `b` (sixteen unsigned 8-bit integer numbers) for which the corresponding bits from target vector `c`
/// (sixteen unsigned 8-bit integer numbers) are 0 and leaves unchanged all destination bits
/// for which the corresponding target bits are 1.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(bmz.v))]
pub unsafe fn __msa_bmz_v(a: v16u8, b: v16u8, c: v16u8) -> v16u8 {
    msa_bmz_v(a, mem::transmute(b), c)
}

/// Immediate Bit Move If Zero
///
/// Copy to destination vector `a` (sixteen unsigned 8-bit integer numbers) all bits from source vector
/// `b` (sixteen unsigned 8-bit integer numbers) for which the corresponding bits from from immediate `imm8`
/// are 0 and leaves unchanged all destination bits for which the corresponding immediate bits are 1.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(bmzi.b, imm8 = 0b11111111))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn __msa_bmzi_b<const IMM8: i32>(a: v16u8, b: v16u8) -> v16u8 {
    static_assert_imm8!(IMM8);
    msa_bmzi_b(a, mem::transmute(b), IMM8)
}

/// Vector Bit Negate
///
/// Negate (complement) one bit in each element of vector `a` (sixteen unsigned 8-bit integer numbers).
/// The bit position is given by the elements in vector `b` (sixteen unsigned 8-bit integer numbers)
/// modulo the size of the element in bits.
/// The result is written to vector (sixteen unsigned 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(bneg.b))]
pub unsafe fn __msa_bneg_b(a: v16u8, b: v16u8) -> v16u8 {
    msa_bneg_b(a, mem::transmute(b))
}

/// Vector Bit Negate
///
/// Negate (complement) one bit in each element of vector `a` (eight unsigned 16-bit integer numbers).
/// The bit position is given by the elements in vector `b` (eight unsigned 16-bit integer numbers)
/// modulo the size of the element in bits.
/// The result is written to vector (eight unsigned 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(bneg.h))]
pub unsafe fn __msa_bneg_h(a: v8u16, b: v8u16) -> v8u16 {
    msa_bneg_h(a, mem::transmute(b))
}

/// Vector Bit Negate
///
/// Negate (complement) one bit in each element of vector `a` (four unsigned 32-bit integer numbers).
/// The bit position is given by the elements in vector `b` (four unsigned 32-bit integer numbers)
/// modulo the size of the element in bits.
/// The result is written to vector (four unsigned 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(bneg.w))]
pub unsafe fn __msa_bneg_w(a: v4u32, b: v4u32) -> v4u32 {
    msa_bneg_w(a, mem::transmute(b))
}

/// Vector Bit Negate
///
/// Negate (complement) one bit in each element of vector `a` (two unsigned 64-bit integer numbers).
/// The bit position is given by the elements in vector `b` (two unsigned 64-bit integer numbers)
/// modulo the size of the element in bits.
/// The result is written to vector (two unsigned 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(bneg.d))]
pub unsafe fn __msa_bneg_d(a: v2u64, b: v2u64) -> v2u64 {
    msa_bneg_d(a, mem::transmute(b))
}

/// Immediate Bit Negate
///
/// Negate (complement) one bit in each element of vector `a` (sixteen unsigned 8-bit integer numbers).
/// The bit position is given by immediate `imm3` modulo the size of the element in bits.
/// The result is written to vector (sixteen unsigned 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(bnegi.b, imm3 = 0b111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_bnegi_b<const IMM3: i32>(a: v16u8) -> v16u8 {
    static_assert_imm3!(IMM3);
    msa_bnegi_b(a, IMM3)
}

/// Immediate Bit Negate
///
/// Negate (complement) one bit in each element of vector `a` (eight unsigned 16-bit integer numbers).
/// The bit position is given by immediate `imm4` modulo the size of the element in bits.
/// The result is written to vector (eight unsigned 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(bnegi.h, imm4 = 0b1111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_bnegi_h<const IMM4: i32>(a: v8u16) -> v8u16 {
    static_assert_imm4!(IMM4);
    msa_bnegi_h(a, IMM4)
}

/// Immediate Bit Negate
///
/// Negate (complement) one bit in each element of vector `a` (four unsigned 32-bit integer numbers).
/// The bit position is given by immediate `imm5` modulo the size of the element in bits.
/// The result is written to vector (four unsigned 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(bnegi.w, imm5 = 0b11111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_bnegi_w<const IMM5: i32>(a: v4u32) -> v4u32 {
    static_assert_imm5!(IMM5);
    msa_bnegi_w(a, IMM5)
}

/// Immediate Bit Negate
///
/// Negate (complement) one bit in each element of vector `a` (two unsigned 64-bit integer numbers).
/// The bit position is given by immediate `imm6` modulo the size of the element in bits.
/// The result is written to vector (two unsigned 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(bnegi.d, imm6 = 0b111111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_bnegi_d<const IMM6: i32>(a: v2u64) -> v2u64 {
    static_assert_imm6!(IMM6);
    msa_bnegi_d(a, IMM6)
}

/// Immediate Branch If All Elements Are Not Zero
///
/// PC-relative branch if all elements in `a` (sixteen unsigned 8-bit integer numbers) are not zero.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(bnz.b))]
pub unsafe fn __msa_bnz_b(a: v16u8) -> i32 {
    msa_bnz_b(a)
}

/// Immediate Branch If All Elements Are Not Zero
///
/// PC-relative branch if all elements in `a` (eight unsigned 16-bit integer numbers) are not zero.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(bnz.h))]
pub unsafe fn __msa_bnz_h(a: v8u16) -> i32 {
    msa_bnz_h(a)
}

/// Immediate Branch If All Elements Are Not Zero
///
/// PC-relative branch if all elements in `a` (four unsigned 32-bit integer numbers) are not zero.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(bnz.w))]
pub unsafe fn __msa_bnz_w(a: v4u32) -> i32 {
    msa_bnz_w(a)
}

/// Immediate Branch If All Elements Are Not Zero
///
/// PC-relative branch if all elements in `a` (two unsigned 64-bit integer numbers) are not zero.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(bnz.d))]
pub unsafe fn __msa_bnz_d(a: v2u64) -> i32 {
    msa_bnz_d(a)
}

/// Immediate Branch If Not Zero (At Least One Element of Any Format Is Not Zero)
///
/// PC-relative branch if at least one bit in `a` (four unsigned 32-bit integer numbers) are not zero.
/// i.e at least one element is not zero regardless of the data format.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(bnz.v))]
pub unsafe fn __msa_bnz_v(a: v16u8) -> i32 {
    msa_bnz_v(a)
}

/// Vector Bit Select
///
/// Selectively copy bits from the source vectors `b` (eight unsigned 16-bit integer numbers)
/// and `c` (eight unsigned 16-bit integer numbers)
/// into destination vector `a` (eight unsigned 16-bit integer numbers) based on the corresponding bit in `a`:
/// if 0 copies the bit from `b`, if 1 copies the bit from `c`.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(bsel.v))]
pub unsafe fn __msa_bsel_v(a: v16u8, b: v16u8, c: v16u8) -> v16u8 {
    msa_bsel_v(a, mem::transmute(b), c)
}

/// Immediate Bit Select
///
/// Selectively copy bits from the 8-bit immediate `imm8` and `c` (eight unsigned 16-bit integer numbers)
/// into destination vector `a` (eight unsigned 16-bit integer numbers) based on the corresponding bit in `a`:
/// if 0 copies the bit from `b`, if 1 copies the bit from `c`.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(bseli.b, imm8 = 0b11111111))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn __msa_bseli_b<const IMM8: i32>(a: v16u8, b: v16u8) -> v16u8 {
    static_assert_imm8!(IMM8);
    msa_bseli_b(a, mem::transmute(b), IMM8)
}

/// Vector Bit Set
///
/// Set to 1 one bit in each element of vector `a` (sixteen unsigned 8-bit integer numbers).
/// The bit position is given by the elements in vector `b` (sixteen unsigned 8-bit integer numbers)
/// modulo the size of the element in bits.
/// The result is written to vector (sixteen unsigned 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(bset.b))]
pub unsafe fn __msa_bset_b(a: v16u8, b: v16u8) -> v16u8 {
    msa_bset_b(a, mem::transmute(b))
}

/// Vector Bit Set
///
/// Set to 1 one bit in each element of vector `a` (eight unsigned 16-bit integer numbers).
/// The bit position is given by the elements in vector `b` (eight unsigned 16-bit integer numbers)
/// modulo the size of the element in bits.
/// The result is written to vector (eight unsigned 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(bset.h))]
pub unsafe fn __msa_bset_h(a: v8u16, b: v8u16) -> v8u16 {
    msa_bset_h(a, mem::transmute(b))
}

/// Vector Bit Set
///
/// Set to 1 one bit in each element of vector `a` (four unsigned 32-bit integer numbers).
/// The bit position is given by the elements in vector `b` (four unsigned 32-bit integer numbers)
/// modulo the size of the element in bits.
/// The result is written to vector (four unsigned 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(bset.w))]
pub unsafe fn __msa_bset_w(a: v4u32, b: v4u32) -> v4u32 {
    msa_bset_w(a, mem::transmute(b))
}

/// Vector Bit Set
///
/// Set to 1 one bit in each element of vector `a` (two unsigned 64-bit integer numbers).
/// The bit position is given by the elements in vector `b` (two unsigned 64-bit integer numbers)
/// modulo the size of the element in bits.
/// The result is written to vector (two unsigned 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(bset.d))]
pub unsafe fn __msa_bset_d(a: v2u64, b: v2u64) -> v2u64 {
    msa_bset_d(a, mem::transmute(b))
}

/// Immediate Bit Set
///
/// Set to 1 one bit in each element of vector `a` (sixteen unsigned 8-bit integer numbers).
/// The bit position is given by immediate `imm3`.
/// The result is written to vector `a` (sixteen unsigned 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(bseti.b, imm3 = 0b111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_bseti_b<const IMM3: i32>(a: v16u8) -> v16u8 {
    static_assert_imm3!(IMM3);
    msa_bseti_b(a, IMM3)
}

/// Immediate Bit Set
///
/// Set to 1 one bit in each element of vector `a` (eight unsigned 16-bit integer numbers).
/// The bit position is given by immediate `imm4`.
/// The result is written to vector `a` (eight unsigned 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(bseti.h, imm4 = 0b1111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_bseti_h<const IMM4: i32>(a: v8u16) -> v8u16 {
    static_assert_imm4!(IMM4);
    msa_bseti_h(a, IMM4)
}

/// Immediate Bit Set
///
/// Set to 1 one bit in each element of vector `a` (four unsigned 32-bit integer numbers).
/// The bit position is given by immediate `imm5`.
/// The result is written to vector `a` (four unsigned 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(bseti.w, imm5 = 0b11111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_bseti_w<const IMM5: i32>(a: v4u32) -> v4u32 {
    static_assert_imm5!(IMM5);
    msa_bseti_w(a, IMM5)
}

/// Immediate Bit Set
///
/// Set to 1 one bit in each element of vector `a` (two unsigned 64-bit integer numbers).
/// The bit position is given by immediate `imm6`.
/// The result is written to vector `a` (two unsigned 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(bseti.d, imm6 = 0b111111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_bseti_d<const IMM6: i32>(a: v2u64) -> v2u64 {
    static_assert_imm6!(IMM6);
    msa_bseti_d(a, IMM6)
}

/// Immediate Branch If At Least One Element Is Zero
///
/// PC-relative branch if at least one element in `a` (sixteen unsigned 8-bit integer numbers) is zero.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(bz.b))]
pub unsafe fn __msa_bz_b(a: v16u8) -> i32 {
    msa_bz_b(a)
}

/// Immediate Branch If At Least One Element Is Zero
///
/// PC-relative branch if at least one element in `a` (eight unsigned 16-bit integer numbers) is zero.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(bz.h))]
pub unsafe fn __msa_bz_h(a: v8u16) -> i32 {
    msa_bz_h(a)
}

/// Immediate Branch If At Least One Element Is Zero
///
/// PC-relative branch if at least one element in `a` (four unsigned 32-bit integer numbers) is zero.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(bz.w))]
pub unsafe fn __msa_bz_w(a: v4u32) -> i32 {
    msa_bz_w(a)
}

/// Immediate Branch If At Least One Element Is Zero
///
/// PC-relative branch if at least one element in `a` (two unsigned 64-bit integer numbers) is zero.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(bz.d))]
pub unsafe fn __msa_bz_d(a: v2u64) -> i32 {
    msa_bz_d(a)
}

/// Immediate Branch If Zero (All Elements of Any Format Are Zero)
///
/// PC-relative branch if all elements in `a` (sixteen unsigned 8-bit integer numbers) bits are zero,
/// i.e. all elements are zero regardless of the data format.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(bz.v))]
pub unsafe fn __msa_bz_v(a: v16u8) -> i32 {
    msa_bz_v(a)
}

/// Vector Compare Equal
///
/// Set all bits to 1 in vector (sixteen signed 8-bit integer numbers) elements
/// if the corresponding `a` (sixteen signed 8-bit integer numbers) and `b` (sixteen signed 8-bit integer numbers)
/// elements are equal, otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ceq.b))]
pub unsafe fn __msa_ceq_b(a: v16i8, b: v16i8) -> v16i8 {
    msa_ceq_b(a, mem::transmute(b))
}

/// Vector Compare Equal
///
/// Set all bits to 1 in vector (eight signed 16-bit integer numbers) elements
/// if the corresponding `a` (eight signed 16-bit integer numbers) and `b` (eight signed 16-bit integer numbers)
/// elements are equal, otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ceq.h))]
pub unsafe fn __msa_ceq_h(a: v8i16, b: v8i16) -> v8i16 {
    msa_ceq_h(a, mem::transmute(b))
}

/// Vector Compare Equal
///
/// Set all bits to 1 in vector (four signed 32-bit integer numbers) elements
/// if the corresponding `a` (four signed 32-bit integer numbers) and `b` (four signed 32-bit integer numbers)
/// elements are equal, otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ceq.w))]
pub unsafe fn __msa_ceq_w(a: v4i32, b: v4i32) -> v4i32 {
    msa_ceq_w(a, mem::transmute(b))
}

/// Vector Compare Equal
///
/// Set all bits to 1 in vector (two signed 64-bit integer numbers) elements
/// if the corresponding `a` (two signed 64-bit integer numbers) and `b` (two signed 64-bit integer numbers)
/// elements are equal, otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ceq.d))]
pub unsafe fn __msa_ceq_d(a: v2i64, b: v2i64) -> v2i64 {
    msa_ceq_d(a, mem::transmute(b))
}

/// Immediate Compare Equal
///
/// Set all bits to 1 in vector (sixteen signed 8-bit integer numbers) elements
/// if the corresponding `a` (sixteen signed 8-bit integer numbers) the 5-bit signed immediate imm_s5
/// are equal, otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ceqi.b, imm_s5 = 0b11111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_ceqi_b<const IMM_S5: i32>(a: v16i8) -> v16i8 {
    static_assert_imm_s5!(IMM_S5);
    msa_ceqi_b(a, IMM_S5)
}

/// Immediate Compare Equal
///
/// Set all bits to 1 in vector (eight signed 16-bit integer numbers) elements
/// if the corresponding `a` (eight signed 16-bit integer numbers) the 5-bit signed immediate imm_s5
/// are equal, otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ceqi.h, imm_s5 = 0b11111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_ceqi_h<const IMM_S5: i32>(a: v8i16) -> v8i16 {
    static_assert_imm_s5!(IMM_S5);
    msa_ceqi_h(a, IMM_S5)
}

/// Immediate Compare Equal
///
/// Set all bits to 1 in vector (four signed 32-bit integer numbers) elements
/// if the corresponding `a` (four signed 32-bit integer numbers) the 5-bit signed immediate imm_s5
/// are equal, otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ceqi.w, imm_s5 = 0b11111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_ceqi_w<const IMM_S5: i32>(a: v4i32) -> v4i32 {
    static_assert_imm_s5!(IMM_S5);
    msa_ceqi_w(a, IMM_S5)
}

/// Immediate Compare Equal
///
/// Set all bits to 1 in vector (two signed 64-bit integer numbers) elements
/// if the corresponding `a` (two signed 64-bit integer numbers) the 5-bit signed immediate imm_s5
/// are equal, otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ceqi.d, imm_s5 = 0b11111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_ceqi_d<const IMM_S5: i32>(a: v2i64) -> v2i64 {
    static_assert_imm_s5!(IMM_S5);
    msa_ceqi_d(a, IMM_S5)
}

/// GPR Copy from MSA Control Register
///
/// The sign extended content of MSA control register cs is copied to GPR rd.
///
/// Can not be tested in user mode
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(cfcmsa, imm5 = 0b11111))]
#[rustc_legacy_const_generics(0)]
pub unsafe fn __msa_cfcmsa<const IMM5: i32>() -> i32 {
    static_assert_imm5!(IMM5);
    msa_cfcmsa(IMM5)
}

/// Vector Compare Signed Less Than or Equal
///
/// Set all bits to 1 in vector (sixteen signed 8-bit integer numbers) elements
/// if the corresponding `a` (sixteen signed 8-bit integer numbers) element
/// are signed less than or equal to `b` (sixteen signed 8-bit integer numbers) element.
/// Otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(cle_s.b))]
pub unsafe fn __msa_cle_s_b(a: v16i8, b: v16i8) -> v16i8 {
    msa_cle_s_b(a, mem::transmute(b))
}

/// Vector Compare Signed Less Than or Equal
///
/// Set all bits to 1 in vector (eight signed 16-bit integer numbers) elements
/// if the corresponding `a` (eight signed 16-bit integer numbers) element
/// are signed less than or equal to `b` (eight signed 16-bit integer numbers) element.
/// Otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(cle_s.h))]
pub unsafe fn __msa_cle_s_h(a: v8i16, b: v8i16) -> v8i16 {
    msa_cle_s_h(a, mem::transmute(b))
}

/// Vector Compare Signed Less Than or Equal
///
/// Set all bits to 1 in vector (four signed 32-bit integer numbers) elements
/// if the corresponding `a` (four signed 32-bit integer numbers) element
/// are signed less than or equal to `b` (four signed 32-bit integer numbers) element.
/// Otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(cle_s.w))]
pub unsafe fn __msa_cle_s_w(a: v4i32, b: v4i32) -> v4i32 {
    msa_cle_s_w(a, mem::transmute(b))
}

/// Vector Compare Signed Less Than or Equal
///
/// Set all bits to 1 in vector (two signed 64-bit integer numbers) elements
/// if the corresponding `a` (two signed 64-bit integer numbers) element
/// are signed less than or equal to `b` (two signed 64-bit integer numbers) element.
/// Otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(cle_s.d))]
pub unsafe fn __msa_cle_s_d(a: v2i64, b: v2i64) -> v2i64 {
    msa_cle_s_d(a, mem::transmute(b))
}

/// Vector Compare Unsigned Less Than or Equal
///
/// Set all bits to 1 in vector (sixteen signed 8-bit integer numbers) elements
/// if the corresponding `a` (sixteen unsigned 8-bit integer numbers) element
/// are unsigned less than or equal to `b` (sixteen unsigned 8-bit integer numbers) element.
/// Otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(cle_u.b))]
pub unsafe fn __msa_cle_u_b(a: v16u8, b: v16u8) -> v16i8 {
    msa_cle_u_b(a, mem::transmute(b))
}

/// Vector Compare Unsigned Less Than or Equal
///
/// Set all bits to 1 in vector (eight signed 16-bit integer numbers) elements
/// if the corresponding `a` (eight unsigned 16-bit integer numbers) element
/// are unsigned less than or equal to `b` (eight unsigned 16-bit integer numbers) element.
/// Otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(cle_u.h))]
pub unsafe fn __msa_cle_u_h(a: v8u16, b: v8u16) -> v8i16 {
    msa_cle_u_h(a, mem::transmute(b))
}

/// Vector Compare Unsigned Less Than or Equal
///
/// Set all bits to 1 in vector (four signed 32-bit integer numbers) elements
/// if the corresponding `a` (four unsigned 32-bit integer numbers) element
/// are unsigned less than or equal to `b` (four unsigned 32-bit integer numbers) element.
/// Otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(cle_u.w))]
pub unsafe fn __msa_cle_u_w(a: v4u32, b: v4u32) -> v4i32 {
    msa_cle_u_w(a, mem::transmute(b))
}

/// Vector Compare Unsigned Less Than or Equal
///
/// Set all bits to 1 in vector (two signed 64-bit integer numbers) elements
/// if the corresponding `a` (two unsigned 64-bit integer numbers) element
/// are unsigned less than or equal to `b` (two unsigned 64-bit integer numbers) element.
/// Otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(cle_u.d))]
pub unsafe fn __msa_cle_u_d(a: v2u64, b: v2u64) -> v2i64 {
    msa_cle_u_d(a, mem::transmute(b))
}

/// Immediate Compare Signed Less Than or Equal
///
/// Set all bits to 1 in vector (sixteen signed 8-bit integer numbers) elements
/// if the corresponding `a` (sixteen signed 8-bit integer numbers) element
/// is less than or equal to the 5-bit signed immediate imm_s5,
/// otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(clei_s.b, imm_s5 = 0b11111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_clei_s_b<const IMM_S5: i32>(a: v16i8) -> v16i8 {
    static_assert_imm_s5!(IMM_S5);
    msa_clei_s_b(a, IMM_S5)
}

/// Immediate Compare Signed Less Than or Equal
///
/// Set all bits to 1 in vector (eight signed 16-bit integer numbers) elements
/// if the corresponding `a` (eight signed 16-bit integer numbers) element
/// is less than or equal to the 5-bit signed immediate imm_s5,
/// otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(clei_s.h, imm_s5 = 0b11111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_clei_s_h<const IMM_S5: i32>(a: v8i16) -> v8i16 {
    static_assert_imm_s5!(IMM_S5);
    msa_clei_s_h(a, IMM_S5)
}

/// Immediate Compare Signed Less Than or Equal
///
/// Set all bits to 1 in vector (four signed 32-bit integer numbers) elements
/// if the corresponding `a` (four signed 32-bit integer numbers) element
/// is less than or equal to the 5-bit signed immediate imm_s5,
/// otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(clei_s.w, imm_s5 = 0b11111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_clei_s_w<const IMM_S5: i32>(a: v4i32) -> v4i32 {
    static_assert_imm_s5!(IMM_S5);
    msa_clei_s_w(a, IMM_S5)
}

/// Immediate Compare Signed Less Than or Equal
///
/// Set all bits to 1 in vector (two signed 64-bit integer numbers) elements
/// if the corresponding `a` (two signed 64-bit integer numbers) element
/// is less than or equal to the 5-bit signed immediate imm_s5,
/// otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(clei_s.d, imm_s5 = 0b11111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_clei_s_d<const IMM_S5: i32>(a: v2i64) -> v2i64 {
    static_assert_imm_s5!(IMM_S5);
    msa_clei_s_d(a, IMM_S5)
}

/// Immediate Compare Unsigned Less Than or Equal
///
/// Set all bits to 1 in vector (sixteen signed 8-bit integer numbers) elements
/// if the corresponding `a` (sixteen unsigned 8-bit integer numbers) element
/// is unsigned less than or equal to the 5-bit unsigned immediate `imm5`,
/// otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(clei_u.b, imm5 = 0b111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_clei_u_b<const IMM5: i32>(a: v16u8) -> v16i8 {
    static_assert_imm5!(IMM5);
    msa_clei_u_b(a, IMM5)
}

/// Immediate Compare Unsigned Less Than or Equal
///
/// Set all bits to 1 in vector (eight signed 16-bit integer numbers) elements
/// if the corresponding `a` (eight unsigned 16-bit integer numbers) element
/// is unsigned less than or equal to the 5-bit unsigned immediate `imm5`,
/// otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(clei_u.h, imm5 = 0b11111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_clei_u_h<const IMM5: i32>(a: v8u16) -> v8i16 {
    static_assert_imm5!(IMM5);
    msa_clei_u_h(a, IMM5)
}

/// Immediate Compare Unsigned Less Than or Equal
///
/// Set all bits to 1 in vector (four signed 32-bit integer numbers) elements
/// if the corresponding `a` (four unsigned 32-bit integer numbers) element
/// is unsigned less than or equal to the 5-bit unsigned immediate `imm5`,
/// otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(clei_u.w, imm5 = 0b11111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_clei_u_w<const IMM5: i32>(a: v4u32) -> v4i32 {
    static_assert_imm5!(IMM5);
    msa_clei_u_w(a, IMM5)
}

/// Immediate Compare Unsigned Less Than or Equal
///
/// Set all bits to 1 in vector (two signed 64-bit integer numbers) elements
/// if the corresponding `a` (two unsigned 64-bit integer numbers) element
/// is unsigned less than or equal to the 5-bit unsigned immediate `imm5`,
/// otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(clei_u.d, imm5 = 0b11111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_clei_u_d<const IMM5: i32>(a: v2u64) -> v2i64 {
    static_assert_imm5!(IMM5);
    msa_clei_u_d(a, IMM5)
}

/// Vector Compare Signed Less Than
///
/// Set all bits to 1 in vector (sixteen signed 8-bit integer numbers) elements
/// if the corresponding `a` (sixteen signed 8-bit integer numbers) element
/// are signed less than `b` (sixteen signed 8-bit integer numbers) element.
/// Otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(clt_s.b))]
pub unsafe fn __msa_clt_s_b(a: v16i8, b: v16i8) -> v16i8 {
    msa_clt_s_b(a, mem::transmute(b))
}

/// Vector Compare Signed Less Than
///
/// Set all bits to 1 in vector (eight signed 16-bit integer numbers) elements
/// if the corresponding `a` (eight signed 16-bit integer numbers) element
/// are signed less than `b` (eight signed 16-bit integer numbers) element.
/// Otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(clt_s.h))]
pub unsafe fn __msa_clt_s_h(a: v8i16, b: v8i16) -> v8i16 {
    msa_clt_s_h(a, mem::transmute(b))
}

/// Vector Compare Signed Less Than
///
/// Set all bits to 1 in vector (four signed 32-bit integer numbers) elements
/// if the corresponding `a` (four signed 32-bit integer numbers) element
/// are signed less than `b` (four signed 32-bit integer numbers) element.
/// Otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(clt_s.w))]
pub unsafe fn __msa_clt_s_w(a: v4i32, b: v4i32) -> v4i32 {
    msa_clt_s_w(a, mem::transmute(b))
}

/// Vector Compare Signed Less Than
///
/// Set all bits to 1 in vector (two signed 64-bit integer numbers) elements
/// if the corresponding `a` (two signed 64-bit integer numbers) element
/// are signed less than `b` (two signed 64-bit integer numbers) element.
/// Otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(clt_s.d))]
pub unsafe fn __msa_clt_s_d(a: v2i64, b: v2i64) -> v2i64 {
    msa_clt_s_d(a, mem::transmute(b))
}

/// Vector Compare Unsigned Less Than
///
/// Set all bits to 1 in vector (sixteen signed 8-bit integer numbers) elements
/// if the corresponding `a` (sixteen unsigned 8-bit integer numbers) element
/// are unsigned less than `b` (sixteen unsigned 8-bit integer numbers) element.
/// Otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(clt_u.b))]
pub unsafe fn __msa_clt_u_b(a: v16u8, b: v16u8) -> v16i8 {
    msa_clt_u_b(a, mem::transmute(b))
}

/// Vector Compare Unsigned Less Than
///
/// Set all bits to 1 in vector (eight signed 16-bit integer numbers) elements
/// if the corresponding `a` (eight unsigned 16-bit integer numbers) element
/// are unsigned less than `b` (eight unsigned 16-bit integer numbers) element.
/// Otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(clt_u.h))]
pub unsafe fn __msa_clt_u_h(a: v8u16, b: v8u16) -> v8i16 {
    msa_clt_u_h(a, mem::transmute(b))
}

/// Vector Compare Unsigned Less Than
///
/// Set all bits to 1 in vector (four signed 32-bit integer numbers) elements
/// if the corresponding `a` (four unsigned 32-bit integer numbers) element
/// are unsigned less than `b` (four unsigned 32-bit integer numbers) element.
/// Otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(clt_u.w))]
pub unsafe fn __msa_clt_u_w(a: v4u32, b: v4u32) -> v4i32 {
    msa_clt_u_w(a, mem::transmute(b))
}

/// Vector Compare Unsigned Less Than
///
/// Set all bits to 1 in vector (two signed 64-bit integer numbers) elements
/// if the corresponding `a` (two unsigned 64-bit integer numbers) element
/// are unsigned less than `b` (two unsigned 64-bit integer numbers) element.
/// Otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(clt_u.d))]
pub unsafe fn __msa_clt_u_d(a: v2u64, b: v2u64) -> v2i64 {
    msa_clt_u_d(a, mem::transmute(b))
}

/// Immediate Compare Signed Less Than
///
/// Set all bits to 1 in vector (sixteen signed 8-bit integer numbers) elements
/// if the corresponding `a` (sixteen signed 8-bit integer numbers) element
/// is less than the 5-bit signed immediate imm_s5,
/// otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(clti_s.b, imm_s5 = 0b111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_clti_s_b<const IMM_S5: i32>(a: v16i8) -> v16i8 {
    static_assert_imm_s5!(IMM_S5);
    msa_clti_s_b(a, IMM_S5)
}

/// Immediate Compare Signed Less Than
///
/// Set all bits to 1 in vector (eight signed 16-bit integer numbers) elements
/// if the corresponding `a` (eight signed 16-bit integer numbers) element
/// is less than the 5-bit signed immediate imm_s5,
/// otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(clti_s.h, imm_s5 = 0b11111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_clti_s_h<const IMM_S5: i32>(a: v8i16) -> v8i16 {
    static_assert_imm_s5!(IMM_S5);
    msa_clti_s_h(a, IMM_S5)
}

/// Immediate Compare Signed Less Than
///
/// Set all bits to 1 in vector (four signed 32-bit integer numbers) elements
/// if the corresponding `a` (four signed 32-bit integer numbers) element
/// is less than the 5-bit signed immediate imm_s5,
/// otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(clti_s.w, imm_s5 = 0b11111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_clti_s_w<const IMM_S5: i32>(a: v4i32) -> v4i32 {
    static_assert_imm_s5!(IMM_S5);
    msa_clti_s_w(a, IMM_S5)
}

/// Immediate Compare Signed Less Than
///
/// Set all bits to 1 in vector (two signed 64-bit integer numbers) elements
/// if the corresponding `a` (two signed 64-bit integer numbers) element
/// is less than the 5-bit signed immediate imm_s5,
/// otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(clti_s.d, imm_s5 = 0b11111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_clti_s_d<const IMM_S5: i32>(a: v2i64) -> v2i64 {
    static_assert_imm_s5!(IMM_S5);
    msa_clti_s_d(a, IMM_S5)
}

/// Immediate Compare Unsigned Less Than
///
/// Set all bits to 1 in vector (sixteen signed 8-bit integer numbers) elements
/// if the corresponding `a` (sixteen unsigned 8-bit integer numbers) element
/// is unsigned less than the 5-bit unsigned immediate `imm5`,
/// otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(clti_u.b, imm5 = 0b111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_clti_u_b<const IMM5: i32>(a: v16u8) -> v16i8 {
    static_assert_imm5!(IMM5);
    msa_clti_u_b(a, IMM5)
}

/// Immediate Compare Unsigned Less Than
///
/// Set all bits to 1 in vector (eight signed 16-bit integer numbers) elements
/// if the corresponding `a` (eight unsigned 16-bit integer numbers) element
/// is unsigned less than the 5-bit unsigned immediate `imm5`,
/// otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(clti_u.h, imm5 = 0b11111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_clti_u_h<const IMM5: i32>(a: v8u16) -> v8i16 {
    static_assert_imm5!(IMM5);
    msa_clti_u_h(a, IMM5)
}

/// Immediate Compare Unsigned Less Than
///
/// Set all bits to 1 in vector (four signed 32-bit integer numbers) elements
/// if the corresponding `a` (four unsigned 32-bit integer numbers) element
/// is unsigned less than the 5-bit unsigned immediate `imm5`,
/// otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(clti_u.w, imm5 = 0b11111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_clti_u_w<const IMM5: i32>(a: v4u32) -> v4i32 {
    static_assert_imm5!(IMM5);
    msa_clti_u_w(a, IMM5)
}

/// Immediate Compare Unsigned Less Than
///
/// Set all bits to 1 in vector (two signed 64-bit integer numbers) elements
/// if the corresponding `a` (two unsigned 64-bit integer numbers) element
/// is unsigned less than the 5-bit unsigned immediate `imm5`,
/// otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(clti_u.d, imm5 = 0b11111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_clti_u_d<const IMM5: i32>(a: v2u64) -> v2i64 {
    static_assert_imm5!(IMM5);
    msa_clti_u_d(a, IMM5)
}

/// Element Copy to GPR Signed
///
/// Sign-extend element `imm4` of vector `a` (sixteen signed 8-bit integer numbers)
/// and copy the result to GPR rd.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(copy_s.b, imm4 = 0b1111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_copy_s_b<const IMM4: i32>(a: v16i8) -> i32 {
    static_assert_imm4!(IMM4);
    msa_copy_s_b(a, IMM4)
}

/// Element Copy to GPR Signed
///
/// Sign-extend element `imm3` of vector `a` (eight signed 16-bit integer numbers)
/// and copy the result to GPR rd.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(copy_s.h, imm3 = 0b111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_copy_s_h<const IMM3: i32>(a: v8i16) -> i32 {
    static_assert_imm3!(IMM3);
    msa_copy_s_h(a, IMM3)
}

/// Element Copy to GPR Signed
///
/// Sign-extend element `imm2` of vector `a` (four signed 32-bit integer numbers)
/// and copy the result to GPR rd.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(copy_s.w, imm2 = 0b11))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_copy_s_w<const IMM2: i32>(a: v4i32) -> i32 {
    static_assert_imm2!(IMM2);
    msa_copy_s_w(a, IMM2)
}

/// Element Copy to GPR Signed
///
/// Sign-extend element `imm1` of vector `a` (two signed 64-bit integer numbers)
/// and copy the result to GPR rd.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(copy_s.d, imm1 = 0b1))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_copy_s_d<const IMM1: i32>(a: v2i64) -> i64 {
    static_assert_imm1!(IMM1);
    msa_copy_s_d(a, IMM1)
}

/// Element Copy to GPR Unsigned
///
/// Zero-extend element `imm4` of vector `a` (sixteen signed 8-bit integer numbers)
/// and copy the result to GPR rd.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(copy_u.b, imm4 = 0b1111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_copy_u_b<const IMM4: i32>(a: v16i8) -> u32 {
    static_assert_imm4!(IMM4);
    msa_copy_u_b(a, IMM4)
}

/// Element Copy to GPR Unsigned
///
/// Zero-extend element `imm3` of vector `a` (eight signed 16-bit integer numbers)
/// and copy the result to GPR rd.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(copy_u.h, imm3 = 0b111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_copy_u_h<const IMM3: i32>(a: v8i16) -> u32 {
    static_assert_imm3!(IMM3);
    msa_copy_u_h(a, IMM3)
}

/// Element Copy to GPR Unsigned
///
/// Zero-extend element `imm2` of vector `a` (four signed 32-bit integer numbers)
/// and copy the result to GPR rd.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(copy_u.w, imm2 = 0b11))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_copy_u_w<const IMM2: i32>(a: v4i32) -> u32 {
    static_assert_imm2!(IMM2);
    msa_copy_u_w(a, IMM2)
}

/// Element Copy to GPR Unsigned
///
/// Zero-extend element `imm1` of vector `a` (two signed 64-bit integer numbers)
/// and copy the result to GPR rd.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(copy_u.d, imm1 = 0b1))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_copy_u_d<const IMM1: i32>(a: v2i64) -> u64 {
    static_assert_imm1!(IMM1);
    msa_copy_u_d(a, IMM1)
}

/// GPR Copy to MSA Control Register
///
/// The content of the least significant 31 bits of GPR `imm1` is copied to
/// MSA control register cd.
///
/// Can not be tested in user mode
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ctcmsa, imm1 = 0b1))]
#[rustc_legacy_const_generics(0)]
pub unsafe fn __msa_ctcmsa<const IMM5: i32>(a: i32) -> () {
    static_assert_imm5!(IMM5);
    msa_ctcmsa(IMM5, a)
}

/// Vector Signed Divide
///
/// The signed integer elements in vector `a` (sixteen signed 8-bit integer numbers)
/// are divided by signed integer elements in vector `b` (sixteen signed 8-bit integer numbers).
/// The result is written to vector (sixteen signed 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(div_s.b))]
pub unsafe fn __msa_div_s_b(a: v16i8, b: v16i8) -> v16i8 {
    msa_div_s_b(a, mem::transmute(b))
}

/// Vector Signed Divide
///
/// The signed integer elements in vector `a` (eight signed 16-bit integer numbers)
/// are divided by signed integer elements in vector `b` (eight signed 16-bit integer numbers).
/// The result is written to vector (eight signed 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(div_s.h))]
pub unsafe fn __msa_div_s_h(a: v8i16, b: v8i16) -> v8i16 {
    msa_div_s_h(a, mem::transmute(b))
}

/// Vector Signed Divide
///
/// The signed integer elements in vector `a` (four signed 32-bit integer numbers)
/// are divided by signed integer elements in vector `b` (four signed 32-bit integer numbers).
/// The result is written to vector (four signed 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(div_s.w))]
pub unsafe fn __msa_div_s_w(a: v4i32, b: v4i32) -> v4i32 {
    msa_div_s_w(a, mem::transmute(b))
}

/// Vector Signed Divide
///
/// The signed integer elements in vector `a` (two signed 64-bit integer numbers)
/// are divided by signed integer elements in vector `b` (two signed 64-bit integer numbers).
/// The result is written to vector (two signed 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(div_s.d))]
pub unsafe fn __msa_div_s_d(a: v2i64, b: v2i64) -> v2i64 {
    msa_div_s_d(a, mem::transmute(b))
}

/// Vector Unsigned Divide
///
/// The unsigned integer elements in vector `a` (sixteen unsigned 8-bit integer numbers)
/// are divided by unsigned integer elements in vector `b` (sixteen unsigned 8-bit integer numbers).
/// The result is written to vector (sixteen unsigned 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(div_u.b))]
pub unsafe fn __msa_div_u_b(a: v16u8, b: v16u8) -> v16u8 {
    msa_div_u_b(a, mem::transmute(b))
}

/// Vector Unsigned Divide
///
/// The unsigned integer elements in vector `a` (eight unsigned 16-bit integer numbers)
/// are divided by unsigned integer elements in vector `b` (eight unsigned 16-bit integer numbers).
/// The result is written to vector (eight unsigned 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(div_u.h))]
pub unsafe fn __msa_div_u_h(a: v8u16, b: v8u16) -> v8u16 {
    msa_div_u_h(a, mem::transmute(b))
}

/// Vector Unsigned Divide
///
/// The unsigned integer elements in vector `a` (four unsigned 32-bit integer numbers)
/// are divided by unsigned integer elements in vector `b` (four unsigned 32-bit integer numbers).
/// The result is written to vector (four unsigned 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(div_u.w))]
pub unsafe fn __msa_div_u_w(a: v4u32, b: v4u32) -> v4u32 {
    msa_div_u_w(a, mem::transmute(b))
}

/// Vector Unsigned Divide
///
/// The unsigned integer elements in vector `a` (two unsigned 64-bit integer numbers)
/// are divided by unsigned integer elements in vector `b` (two unsigned 64-bit integer numbers).
/// The result is written to vector (two unsigned 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(div_u.d))]
pub unsafe fn __msa_div_u_d(a: v2u64, b: v2u64) -> v2u64 {
    msa_div_u_d(a, mem::transmute(b))
}

/// Vector Signed Dot Product
///
/// The signed integer elements in vector `a` (sixteen signed 8-bit integer numbers)
/// are multiplied by signed integer elements in vector `b` (sixteen signed 8-bit integer numbers)
/// producing a result the size of the input operands. The multiplication results of
/// adjacent odd/even elements are added and stored to the destination
/// vector (eight signed 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(dotp_s.h))]
pub unsafe fn __msa_dotp_s_h(a: v16i8, b: v16i8) -> v8i16 {
    msa_dotp_s_h(a, mem::transmute(b))
}

/// Vector Signed Dot Product
///
/// The signed integer elements in vector `a` (eight signed 16-bit integer numbers)
/// are multiplied by signed integer elements in vector `b` (eight signed 16-bit integer numbers)
/// producing a result the size of the input operands. The multiplication results of
/// adjacent odd/even elements are added and stored to the destination
/// vector (four signed 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(dotp_s.w))]
pub unsafe fn __msa_dotp_s_w(a: v8i16, b: v8i16) -> v4i32 {
    msa_dotp_s_w(a, mem::transmute(b))
}

/// Vector Signed Dot Product
///
/// The signed integer elements in vector `a` (four signed 32-bit integer numbers)
/// are multiplied by signed integer elements in vector `b` (four signed 32-bit integer numbers)
/// producing a result the size of the input operands. The multiplication results of
/// adjacent odd/even elements are added and stored to the destination
/// vector (two signed 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(dotp_s.d))]
pub unsafe fn __msa_dotp_s_d(a: v4i32, b: v4i32) -> v2i64 {
    msa_dotp_s_d(a, mem::transmute(b))
}

/// Vector Unsigned Dot Product
///
/// The unsigned integer elements in vector `a` (sixteen unsigned 8-bit integer numbers)
/// are multiplied by unsigned integer elements in vector `b` (sixteen unsigned 8-bit integer numbers)
/// producing a result the size of the input operands. The multiplication results of
/// adjacent odd/even elements are added and stored to the destination
/// vector (eight unsigned 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(dotp_u.h))]
pub unsafe fn __msa_dotp_u_h(a: v16u8, b: v16u8) -> v8u16 {
    msa_dotp_u_h(a, mem::transmute(b))
}

/// Vector Unsigned Dot Product
///
/// The unsigned integer elements in vector `a` (eight unsigned 16-bit integer numbers)
/// are multiplied by unsigned integer elements in vector `b` (eight unsigned 16-bit integer numbers)
/// producing a result the size of the input operands. The multiplication results of
/// adjacent odd/even elements are added and stored to the destination
/// vector (four unsigned 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(dotp_u.w))]
pub unsafe fn __msa_dotp_u_w(a: v8u16, b: v8u16) -> v4u32 {
    msa_dotp_u_w(a, mem::transmute(b))
}

/// Vector Unsigned Dot Product
///
/// The unsigned integer elements in vector `a` (four unsigned 32-bit integer numbers)
/// are multiplied by unsigned integer elements in vector `b` (four unsigned 32-bit integer numbers)
/// producing a result the size of the input operands. The multiplication results of
/// adjacent odd/even elements are added and stored to the destination
/// vector (two unsigned 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(dotp_u.d))]
pub unsafe fn __msa_dotp_u_d(a: v4u32, b: v4u32) -> v2u64 {
    msa_dotp_u_d(a, mem::transmute(b))
}

/// Vector Signed Dot Product and Add
///
/// The signed integer elements in vector `b` (sixteen signed 8-bit integer numbers)
/// are multiplied by signed integer elements in vector `c` (sixteen signed 8-bit integer numbers)
/// producing a result twice the size of the input operands. The multiplication results
/// of adjacent odd/even elements are added to the vector `a` (eight signed 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(dpadd_s.h))]
pub unsafe fn __msa_dpadd_s_h(a: v8i16, b: v16i8, c: v16i8) -> v8i16 {
    msa_dpadd_s_h(a, mem::transmute(b), c)
}

/// Vector Signed Dot Product and Add
///
/// The signed integer elements in vector `b` (eight signed 16-bit integer numbers)
/// are multiplied by signed integer elements in vector `c` (eight signed 16-bit integer numbers)
/// producing a result twice the size of the input operands. The multiplication results
/// of adjacent odd/even elements are added to the vector `a` (four signed 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(dpadd_s.w))]
pub unsafe fn __msa_dpadd_s_w(a: v4i32, b: v8i16, c: v8i16) -> v4i32 {
    msa_dpadd_s_w(a, mem::transmute(b), c)
}

/// Vector Signed Dot Product and Add
///
/// The signed integer elements in vector `b` (four signed 32-bit integer numbers)
/// are multiplied by signed integer elements in vector `c` (four signed 32-bit integer numbers)
/// producing a result twice the size of the input operands. The multiplication results
/// of adjacent odd/even elements are added to the vector `a` (two signed 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(dpadd_s.d))]
pub unsafe fn __msa_dpadd_s_d(a: v2i64, b: v4i32, c: v4i32) -> v2i64 {
    msa_dpadd_s_d(a, mem::transmute(b), c)
}

/// Vector Unsigned Dot Product and Add
///
/// The unsigned integer elements in vector `b` (sixteen unsigned 8-bit integer numbers)
/// are multiplied by unsigned integer elements in vector `c` (sixteen unsigned 8-bit integer numbers)
/// producing a result twice the size of the input operands. The multiplication results
/// of adjacent odd/even elements are added to the vector `a` (eight unsigned 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(dpadd_u.h))]
pub unsafe fn __msa_dpadd_u_h(a: v8u16, b: v16u8, c: v16u8) -> v8u16 {
    msa_dpadd_u_h(a, mem::transmute(b), c)
}

/// Vector Unsigned Dot Product and Add
///
/// The unsigned integer elements in vector `b` (eight unsigned 16-bit integer numbers)
/// are multiplied by unsigned integer elements in vector `c` (eight unsigned 16-bit integer numbers)
/// producing a result twice the size of the input operands. The multiplication results
/// of adjacent odd/even elements are added to the vector `a` (four unsigned 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(dpadd_u.w))]
pub unsafe fn __msa_dpadd_u_w(a: v4u32, b: v8u16, c: v8u16) -> v4u32 {
    msa_dpadd_u_w(a, mem::transmute(b), c)
}

/// Vector Unsigned Dot Product and Add
///
/// The unsigned integer elements in vector `b` (four unsigned 32-bit integer numbers)
/// are multiplied by unsigned integer elements in vector `c` (four unsigned 32-bit integer numbers)
/// producing a result twice the size of the input operands. The multiplication results
/// of adjacent odd/even elements are added to the vector `a` (two unsigned 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(dpadd_u.d))]
pub unsafe fn __msa_dpadd_u_d(a: v2u64, b: v4u32, c: v4u32) -> v2u64 {
    msa_dpadd_u_d(a, mem::transmute(b), c)
}

/// Vector Signed Dot Product and Add
///
/// The signed integer elements in vector `b` (sixteen signed 8-bit integer numbers)
/// are multiplied by signed integer elements in vector `c` (sixteen signed 8-bit integer numbers)
/// producing a result twice the size of the input operands. The multiplication results
/// of adjacent odd/even elements are subtracted from the integer elements in vector `a`
/// (eight signed 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(dpsub_s.h))]
pub unsafe fn __msa_dpsub_s_h(a: v8i16, b: v16i8, c: v16i8) -> v8i16 {
    msa_dpsub_s_h(a, mem::transmute(b), c)
}

/// Vector Signed Dot Product and Add
///
/// The signed integer elements in vector `b` (eight signed 16-bit integer numbers)
/// are multiplied by signed integer elements in vector `c` (eight signed 16-bit integer numbers)
/// producing a result twice the size of the input operands. The multiplication results
/// of adjacent odd/even elements are subtracted from the integer elements in vector `a`
/// (four signed 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(dpsub_s.w))]
pub unsafe fn __msa_dpsub_s_w(a: v4i32, b: v8i16, c: v8i16) -> v4i32 {
    msa_dpsub_s_w(a, mem::transmute(b), c)
}

/// Vector Signed Dot Product and Add
///
/// The signed integer elements in vector `b` (four signed 32-bit integer numbers)
/// are multiplied by signed integer elements in vector `c` (four signed 32-bit integer numbers)
/// producing a result twice the size of the input operands. The multiplication results
/// of adjacent odd/even elements are subtracted from the integer elements in vector `a`
/// (two signed 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(dpsub_s.d))]
pub unsafe fn __msa_dpsub_s_d(a: v2i64, b: v4i32, c: v4i32) -> v2i64 {
    msa_dpsub_s_d(a, mem::transmute(b), c)
}

/// Vector Unsigned Dot Product and Add
///
/// The unsigned integer elements in vector `b` (sixteen unsigned 8-bit integer numbers)
/// are multiplied by unsigned integer elements in vector `c` (sixteen unsigned 8-bit integer numbers)
/// producing a result twice the size of the input operands. The multiplication results
/// of adjacent odd/even elements are subtracted from the integer elements in vector `a`
/// (eight signed 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(dpsub_u.h))]
pub unsafe fn __msa_dpsub_u_h(a: v8i16, b: v16u8, c: v16u8) -> v8i16 {
    msa_dpsub_u_h(a, mem::transmute(b), c)
}

/// Vector Unsigned Dot Product and Add
///
/// The unsigned integer elements in vector `b` (eight unsigned 16-bit integer numbers)
/// are multiplied by unsigned integer elements in vector `c` (eight unsigned 16-bit integer numbers)
/// producing a result twice the size of the input operands. The multiplication results
/// of adjacent odd/even elements are subtracted from the integer elements in vector `a`
/// (four signed 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(dpsub_u.w))]
pub unsafe fn __msa_dpsub_u_w(a: v4i32, b: v8u16, c: v8u16) -> v4i32 {
    msa_dpsub_u_w(a, mem::transmute(b), c)
}

/// Vector Unsigned Dot Product and Add
///
/// The unsigned integer elements in vector `b` (four unsigned 32-bit integer numbers)
/// are multiplied by unsigned integer elements in vector `c` (four unsigned 32-bit integer numbers)
/// producing a result twice the size of the input operands. The multiplication results
/// of adjacent odd/even elements are subtracted from the integer elements in vector `a`
/// (two signed 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(dpsub_u.d))]
pub unsafe fn __msa_dpsub_u_d(a: v2i64, b: v4u32, c: v4u32) -> v2i64 {
    msa_dpsub_u_d(a, mem::transmute(b), c)
}

/// Vector Floating-Point Addition
///
/// The floating-point elements in vector `a` (four 32-bit floating point numbers)
/// are added to the floating-point elements in `bc` (four 32-bit floating point numbers).
/// The result is written to vector (four 32-bit floating point numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fadd.w))]
pub unsafe fn __msa_fadd_w(a: v4f32, b: v4f32) -> v4f32 {
    msa_fadd_w(a, mem::transmute(b))
}

/// Vector Floating-Point Addition
///
/// The floating-point elements in vector `a` (two 64-bit floating point numbers)
/// are added to the floating-point elements in `bc` (two 64-bit floating point numbers).
/// The result is written to vector (two 64-bit floating point numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fadd.d))]
pub unsafe fn __msa_fadd_d(a: v2f64, b: v2f64) -> v2f64 {
    msa_fadd_d(a, mem::transmute(b))
}

/// Vector Floating-Point Quiet Compare Always False
///
/// Set all bits to 0 in vector (four signed 32-bit integer numbers).
/// Signaling NaN elements in `a` (four 32-bit floating point numbers)
/// or `b` (four 32-bit floating point numbers) signal Invalid Operation exception.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fcaf.w))]
pub unsafe fn __msa_fcaf_w(a: v4f32, b: v4f32) -> v4i32 {
    msa_fcaf_w(a, mem::transmute(b))
}

/// Vector Floating-Point Quiet Compare Always False
///
/// Set all bits to 0 in vector (two signed 64-bit integer numbers).
/// Signaling NaN elements in `a` (two 64-bit floating point numbers)
/// or `b` (two 64-bit floating point numbers) signal Invalid Operation exception.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fcaf.d))]
pub unsafe fn __msa_fcaf_d(a: v2f64, b: v2f64) -> v2i64 {
    msa_fcaf_d(a, mem::transmute(b))
}

/// Vector Floating-Point Quiet Compare Equal
///
/// Set all bits to 1 in vector (four signed 32-bit integer numbers)
/// elements if the corresponding in `a` (four 32-bit floating point numbers)
/// and `b` (four 32-bit floating point numbers) elements are ordered and equal,
/// otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fceq.w))]
pub unsafe fn __msa_fceq_w(a: v4f32, b: v4f32) -> v4i32 {
    msa_fceq_w(a, mem::transmute(b))
}

/// Vector Floating-Point Quiet Compare Equal
///
/// Set all bits to 1 in vector (two signed 64-bit integer numbers)
/// elements if the corresponding in `a` (two 64-bit floating point numbers)
/// and `b` (two 64-bit floating point numbers) elements are ordered and equal,
/// otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fceq.d))]
pub unsafe fn __msa_fceq_d(a: v2f64, b: v2f64) -> v2i64 {
    msa_fceq_d(a, mem::transmute(b))
}

/// Vector Floating-Point Class Mask
///
/// Store in each element of vector (four signed 32-bit integer numbers)
/// a bit mask reflecting the floating-point class of the corresponding element of vector
/// `a` (four 32-bit floating point numbers).
/// The mask has 10 bits as follows. Bits 0 and 1 indicate NaN values: signaling NaN (bit 0) and quiet NaN (bit 1).
/// Bits 2, 3, 4, 5 classify negative values: infinity (bit 2), normal (bit 3), subnormal (bit 4), and zero (bit 5).
/// Bits 6, 7, 8, 9 classify positive values: infinity (bit 6), normal (bit 7), subnormal (bit 8), and zero (bit 9).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fclass.w))]
pub unsafe fn __msa_fclass_w(a: v4f32) -> v4i32 {
    msa_fclass_w(a)
}

/// Vector Floating-Point Class Mask
///
/// Store in each element of vector (two signed 64-bit integer numbers)
/// a bit mask reflecting the floating-point class of the corresponding element of vector
/// `a` (two 64-bit floating point numbers).
/// The mask has 10 bits as follows. Bits 0 and 1 indicate NaN values: signaling NaN (bit 0) and quiet NaN (bit 1).
/// Bits 2, 3, 4, 5 classify negative values: infinity (bit 2), normal (bit 3), subnormal (bit 4), and zero (bit 5).
/// Bits 6, 7, 8, 9 classify positive values: infinity (bit 6), normal (bit 7), subnormal (bit 8), and zero (bit 9).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fclass.d))]
pub unsafe fn __msa_fclass_d(a: v2f64) -> v2i64 {
    msa_fclass_d(a)
}

/// Vector Floating-Point Quiet Compare Less or Equal
///
/// Set all bits to 1 in vector (four signed 32-bit integer numbers)
/// elements if the corresponding `a` (four 32-bit floating point numbers) elements are ordered
/// and either less than or equal to `b` (four 32-bit floating point numbers) elements,
/// otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fcle.w))]
pub unsafe fn __msa_fcle_w(a: v4f32, b: v4f32) -> v4i32 {
    msa_fcle_w(a, mem::transmute(b))
}

/// Vector Floating-Point Quiet Compare Less or Equal
///
/// Set all bits to 1 in vector (two signed 64-bit integer numbers)
/// elements if the corresponding `a` (two 64-bit floating point numbers) elements are ordered
/// and either less than or equal to `b` (two 64-bit floating point numbers) elements,
/// otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fcle.d))]
pub unsafe fn __msa_fcle_d(a: v2f64, b: v2f64) -> v2i64 {
    msa_fcle_d(a, mem::transmute(b))
}

/// Vector Floating-Point Quiet Compare Less Than
///
/// Set all bits to 1 in vector (four signed 32-bit integer numbers)
/// elements if the corresponding `a` (four 32-bit floating point numbers) elements are ordered
/// and less than `b` (four 32-bit floating point numbers) elements,
/// otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fclt.w))]
pub unsafe fn __msa_fclt_w(a: v4f32, b: v4f32) -> v4i32 {
    msa_fclt_w(a, mem::transmute(b))
}

/// Vector Floating-Point Quiet Compare Less Than
///
/// Set all bits to 1 in vector (two signed 64-bit integer numbers)
/// elements if the corresponding `a` (two 64-bit floating point numbers) elements are ordered
/// and less than `b` (two 64-bit floating point numbers) elements,
/// otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fclt.d))]
pub unsafe fn __msa_fclt_d(a: v2f64, b: v2f64) -> v2i64 {
    msa_fclt_d(a, mem::transmute(b))
}

/// Vector Floating-Point Quiet Compare Not Equal
///
/// Set all bits to 1 in vector (four signed 32-bit integer numbers)
/// elements if the corresponding `a` (four 32-bit floating point numbers) and
/// `b` (four 32-bit floating point numbers) elements are ordered and not equal,
/// otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fcne.w))]
pub unsafe fn __msa_fcne_w(a: v4f32, b: v4f32) -> v4i32 {
    msa_fcne_w(a, mem::transmute(b))
}

/// Vector Floating-Point Quiet Compare Not Equal
///
/// Set all bits to 1 in vector (two signed 64-bit integer numbers)
/// elements if the corresponding `a` (two 64-bit floating point numbers) and
/// `b` (two 64-bit floating point numbers) elements are ordered and not equal,
/// otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fcne.d))]
pub unsafe fn __msa_fcne_d(a: v2f64, b: v2f64) -> v2i64 {
    msa_fcne_d(a, mem::transmute(b))
}

/// Vector Floating-Point Quiet Compare Ordered
///
/// Set all bits to 1 in vector (four signed 32-bit integer numbers)
/// elements if the corresponding `a` (four 32-bit floating point numbers) and
/// `b` (four 32-bit floating point numbers) elements are ordered, i.e. both elements are not NaN values,
/// otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fcor.w))]
pub unsafe fn __msa_fcor_w(a: v4f32, b: v4f32) -> v4i32 {
    msa_fcor_w(a, mem::transmute(b))
}

/// Vector Floating-Point Quiet Compare Ordered
///
/// Set all bits to 1 in vector (two signed 64-bit integer numbers)
/// elements if the corresponding `a` (two 64-bit floating point numbers) and
/// `b` (two 64-bit floating point numbers) elements are ordered, i.e. both elements are not NaN values,
/// otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fcor.d))]
pub unsafe fn __msa_fcor_d(a: v2f64, b: v2f64) -> v2i64 {
    msa_fcor_d(a, mem::transmute(b))
}

/// Vector Floating-Point Quiet Compare Unordered or Equal
///
/// Set all bits to 1 in vector (four signed 32-bit integer numbers)
/// elements if the corresponding `a` (four 32-bit floating point numbers) and
/// `b` (four 32-bit floating point numbers) elements are unordered or equal,
/// otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fcueq.w))]
pub unsafe fn __msa_fcueq_w(a: v4f32, b: v4f32) -> v4i32 {
    msa_fcueq_w(a, mem::transmute(b))
}

/// Vector Floating-Point Quiet Compare Unordered or Equal
///
/// Set all bits to 1 in vector (two signed 64-bit integer numbers)
/// elements if the corresponding `a` (two 64-bit floating point numbers) and
/// `b` (two 64-bit floating point numbers) elements are unordered or equal,
/// otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fcueq.d))]
pub unsafe fn __msa_fcueq_d(a: v2f64, b: v2f64) -> v2i64 {
    msa_fcueq_d(a, mem::transmute(b))
}

/// Vector Floating-Point Quiet Compare Unordered or Less or Equal
///
/// Set all bits to 1 in vector (four signed 32-bit integer numbers)
/// elements if the corresponding elements in `a` (four 32-bit floating point numbers)
/// are unordered or less than or equal to `b` (four 32-bit floating point numbers) elements,
/// otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fcule.w))]
pub unsafe fn __msa_fcule_w(a: v4f32, b: v4f32) -> v4i32 {
    msa_fcule_w(a, mem::transmute(b))
}

/// Vector Floating-Point Quiet Compare Unordered or Less or Equal
///
/// Set all bits to 1 in vector (two signed 64-bit integer numbers)
/// elements if the corresponding elements in `a` (two 64-bit floating point numbers)
/// are unordered or less than or equal to `b` (two 64-bit floating point numbers) elements,
/// otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fcule.d))]
pub unsafe fn __msa_fcule_d(a: v2f64, b: v2f64) -> v2i64 {
    msa_fcule_d(a, mem::transmute(b))
}

/// Vector Floating-Point Quiet Compare Unordered or Less Than
///
/// Set all bits to 1 in vector (four signed 32-bit integer numbers)
/// elements if the corresponding elements in `a` (four 32-bit floating point numbers)
/// are unordered or less than `b` (four 32-bit floating point numbers) elements,
/// otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fcult.w))]
pub unsafe fn __msa_fcult_w(a: v4f32, b: v4f32) -> v4i32 {
    msa_fcult_w(a, mem::transmute(b))
}

/// Vector Floating-Point Quiet Compare Unordered or Less Than
///
/// Set all bits to 1 in vector (two signed 64-bit integer numbers)
/// elements if the corresponding elements in `a` (two 64-bit floating point numbers)
/// are unordered or less than `b` (two 64-bit floating point numbers) elements,
/// otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fcult.d))]
pub unsafe fn __msa_fcult_d(a: v2f64, b: v2f64) -> v2i64 {
    msa_fcult_d(a, mem::transmute(b))
}

/// Vector Floating-Point Quiet Compare Unordered
///
/// Set all bits to 1 in vector (four signed 32-bit integer numbers)
/// elements if the corresponding `a` (four 32-bit floating point numbers)
/// and `b` (four 32-bit floating point numbers) elements are unordered,
/// i.e. at least one element is a NaN value, otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fcun.w))]
pub unsafe fn __msa_fcun_w(a: v4f32, b: v4f32) -> v4i32 {
    msa_fcun_w(a, mem::transmute(b))
}

/// Vector Floating-Point Quiet Compare Unordered
///
/// Set all bits to 1 in vector (two signed 64-bit integer numbers)
/// elements if the corresponding `a` (two 64-bit floating point numbers)
/// and `b` (two 64-bit floating point numbers) elements are unordered,
/// i.e. at least one element is a NaN value, otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fcun.d))]
pub unsafe fn __msa_fcun_d(a: v2f64, b: v2f64) -> v2i64 {
    msa_fcun_d(a, mem::transmute(b))
}

/// Vector Floating-Point Quiet Compare Unordered or Not Equal
///
/// Set all bits to 1 in vector (four signed 32-bit integer numbers)
/// elements if the corresponding `a` (four 32-bit floating point numbers)
/// and `b` (four 32-bit floating point numbers) elements are unordered or not equal,
/// otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fcune.w))]
pub unsafe fn __msa_fcune_w(a: v4f32, b: v4f32) -> v4i32 {
    msa_fcune_w(a, mem::transmute(b))
}

/// Vector Floating-Point Quiet Compare Unordered or Not Equal
///
/// Set all bits to 1 in vector (two signed 64-bit integer numbers)
/// elements if the corresponding `a` (two 64-bit floating point numbers)
/// and `b` (two 64-bit floating point numbers) elements are unordered or not equal,
/// otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fcune.d))]
pub unsafe fn __msa_fcune_d(a: v2f64, b: v2f64) -> v2i64 {
    msa_fcune_d(a, mem::transmute(b))
}

/// Vector Floating-Point Division
///
/// The floating-point elements in vector `a` (four 32-bit floating point numbers)
/// are divided by the floating-point elements in vector `b` (four 32-bit floating point numbers).
/// The result is written to vector (four 32-bit floating point numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fdiv.w))]
pub unsafe fn __msa_fdiv_w(a: v4f32, b: v4f32) -> v4f32 {
    msa_fdiv_w(a, mem::transmute(b))
}

/// Vector Floating-Point Division
///
/// The floating-point elements in vector `a` (two 64-bit floating point numbers)
/// are divided by the floating-point elements in vector `b` (two 64-bit floating point numbers).
/// The result is written to vector (two 64-bit floating point numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fdiv.d))]
pub unsafe fn __msa_fdiv_d(a: v2f64, b: v2f64) -> v2f64 {
    msa_fdiv_d(a, mem::transmute(b))
}

/* FIXME: 16-bit float
/// Vector Floating-Point Down-Convert Interchange Format
///
/// The floating-point elements in vector `a` (four 64-bit floating point numbers)
/// and vector `b` (four 64-bit floating point numbers) are down-converted
/// to a smaller interchange format, i.e. from 64-bit to 32-bit, or from 32-bit to 16-bit.
/// The result is written to vector (8 16-bit floating point numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fexdo.h))]
pub unsafe fn __msa_fexdo_h(a: v4f32, b: v4f32) -> f16x8 {
    msa_fexdo_h(a, mem::transmute(b))
}*/

/// Vector Floating-Point Down-Convert Interchange Format
///
/// The floating-point elements in vector `a` (two 64-bit floating point numbers)
/// and vector `b` (two 64-bit floating point numbers) are down-converted
/// to a smaller interchange format, i.e. from 64-bit to 32-bit, or from 32-bit to 16-bit.
/// The result is written to vector (four 32-bit floating point numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fexdo.w))]
pub unsafe fn __msa_fexdo_w(a: v2f64, b: v2f64) -> v4f32 {
    msa_fexdo_w(a, mem::transmute(b))
}

/// Vector Floating-Point Down-Convert Interchange Format
///
/// The floating-point elements in vector `a` (four 32-bit floating point numbers)
/// are scaled, i.e. multiplied, by 2 to the power of integer elements in vector `b`
/// (four signed 32-bit integer numbers).
/// The result is written to vector (four 32-bit floating point numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fexp2.w))]
pub unsafe fn __msa_fexp2_w(a: v4f32, b: v4i32) -> v4f32 {
    msa_fexp2_w(a, mem::transmute(b))
}

/// Vector Floating-Point Down-Convert Interchange Format
///
/// The floating-point elements in vector `a` (two 64-bit floating point numbers)
/// are scaled, i.e. multiplied, by 2 to the power of integer elements in vector `b`
/// (two signed 64-bit integer numbers).
/// The result is written to vector (two 64-bit floating point numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fexp2.d))]
pub unsafe fn __msa_fexp2_d(a: v2f64, b: v2i64) -> v2f64 {
    msa_fexp2_d(a, mem::transmute(b))
}

/* FIXME: 16-bit float
/// Vector Floating-Point Up-Convert Interchange Format Left
///
/// The left half floating-point elements in vector `a` (two 16-bit floating point numbers)
/// are up-converted to a larger interchange format,
/// i.e. from 16-bit to 32-bit, or from 32-bit to 64-bit.
/// The result is written to vector (four 32-bit floating point numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fexupl.w))]
pub unsafe fn __msa_fexupl_w(a: f16x8) -> v4f32 {
    msa_fexupl_w(a)
}*/

/// Vector Floating-Point Up-Convert Interchange Format Left
///
/// The left half floating-point elements in vector `a` (four 32-bit floating point numbers)
/// are up-converted to a larger interchange format,
/// i.e. from 16-bit to 32-bit, or from 32-bit to 64-bit.
/// The result is written to vector (two 64-bit floating point numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fexupl.d))]
pub unsafe fn __msa_fexupl_d(a: v4f32) -> v2f64 {
    msa_fexupl_d(a)
}

/* FIXME: 16-bit float
/// Vector Floating-Point Up-Convert Interchange Format Left
///
/// The right half floating-point elements in vector `a` (two 16-bit floating point numbers)
/// are up-converted to a larger interchange format,
/// i.e. from 16-bit to 32-bit, or from 32-bit to 64-bit.
/// The result is written to vector (four 32-bit floating point numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fexupr.w))]
pub unsafe fn __msa_fexupr_w(a: f16x8) -> v4f32 {
    msa_fexupr_w(a)
} */

/// Vector Floating-Point Up-Convert Interchange Format Left
///
/// The right half floating-point elements in vector `a` (four 32-bit floating point numbers)
/// are up-converted to a larger interchange format,
/// i.e. from 16-bit to 32-bit, or from 32-bit to 64-bit.
/// The result is written to vector (two 64-bit floating point numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fexupr.d))]
pub unsafe fn __msa_fexupr_d(a: v4f32) -> v2f64 {
    msa_fexupr_d(a)
}

/// Vector Floating-Point Round and Convert from Signed Integer
///
/// The signed integer elements in vector `a` (four signed 32-bit integer numbers)
/// are converted to floating-point values.
/// The result is written to vector (four 32-bit floating point numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ffint_s.w))]
pub unsafe fn __msa_ffint_s_w(a: v4i32) -> v4f32 {
    msa_ffint_s_w(a)
}

/// Vector Floating-Point Round and Convert from Signed Integer
///
/// The signed integer elements in vector `a` (two signed 64-bit integer numbers)
/// are converted to floating-point values.
/// The result is written to vector (two 64-bit floating point numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ffint_s.d))]
pub unsafe fn __msa_ffint_s_d(a: v2i64) -> v2f64 {
    msa_ffint_s_d(a)
}

/// Vector Floating-Point Round and Convert from Unsigned Integer
///
/// The unsigned integer elements in vector `a` (four unsigned 32-bit integer numbers)
/// are converted to floating-point values.
/// The result is written to vector (four 32-bit floating point numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ffint_u.w))]
pub unsafe fn __msa_ffint_u_w(a: v4u32) -> v4f32 {
    msa_ffint_u_w(a)
}

/// Vector Floating-Point Round and Convert from Unsigned Integer
///
/// The unsigned integer elements in vector `a` (two unsigned 64-bit integer numbers)
/// are converted to floating-point values.
/// The result is written to vector (two 64-bit floating point numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ffint_u.d))]
pub unsafe fn __msa_ffint_u_d(a: v2u64) -> v2f64 {
    msa_ffint_u_d(a)
}

/// Vector Floating-Point Convert from Fixed-Point Left
///
/// The left half fixed-point elements in vector `a` (eight signed 16-bit integer numbers)
/// are up-converted to floating-point data format.
/// i.e. from 16-bit Q15 to 32-bit floating-point, or from 32-bit Q31 to 64-bit floating-point.
/// The result is written to vector (four 32-bit floating point numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ffql.w))]
pub unsafe fn __msa_ffql_w(a: v8i16) -> v4f32 {
    msa_ffql_w(a)
}

/// Vector Floating-Point Convert from Fixed-Point Left
///
/// The left half fixed-point elements in vector `a` (four signed 32-bit integer numbers)
/// are up-converted to floating-point data format.
/// i.e. from 16-bit Q15 to 32-bit floating-point, or from 32-bit Q31 to 64-bit floating-point.
/// The result is written to vector (two 64-bit floating point numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ffql.d))]
pub unsafe fn __msa_ffql_d(a: v4i32) -> v2f64 {
    msa_ffql_d(a)
}

/// Vector Floating-Point Convert from Fixed-Point Left
///
/// The right half fixed-point elements in vector `a` (eight signed 16-bit integer numbers)
/// are up-converted to floating-point data format.
/// i.e. from 16-bit Q15 to 32-bit floating-point, or from 32-bit Q31 to 64-bit floating-point.
/// The result is written to vector (four 32-bit floating point numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ffqr.w))]
pub unsafe fn __msa_ffqr_w(a: v8i16) -> v4f32 {
    msa_ffqr_w(a)
}

/// Vector Floating-Point Convert from Fixed-Point Left
///
/// The right half fixed-point elements in vector `a` (four signed 32-bit integer numbers)
/// are up-converted to floating-point data format.
/// i.e. from 16-bit Q15 to 32-bit floating-point, or from 32-bit Q31 to 64-bit floating-point.
/// The result is written to vector (two 64-bit floating point numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ffqr.d))]
pub unsafe fn __msa_ffqr_d(a: v4i32) -> v2f64 {
    msa_ffqr_d(a)
}

/// Vector Fill from GPR
///
/// Replicate GPR rs value to all elements in vector (sixteen signed 8-bit integer numbers).
/// If the source GPR is wider than the destination data format, the destination's elements
/// will be set to the least significant bits of the GPR.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fill.b))]
pub unsafe fn __msa_fill_b(a: i32) -> v16i8 {
    msa_fill_b(a)
}

/// Vector Fill from GPR
///
/// Replicate GPR rs value to all elements in vector (eight signed 16-bit integer numbers).
/// If the source GPR is wider than the destination data format, the destination's elements
/// will be set to the least significant bits of the GPR.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fill.h))]
pub unsafe fn __msa_fill_h(a: i32) -> v8i16 {
    msa_fill_h(a)
}

/// Vector Fill from GPR
///
/// Replicate GPR rs value to all elements in vector (four signed 32-bit integer numbers).
/// If the source GPR is wider than the destination data format, the destination's elements
/// will be set to the least significant bits of the GPR.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fill.w))]
pub unsafe fn __msa_fill_w(a: i32) -> v4i32 {
    msa_fill_w(a)
}

/// Vector Fill from GPR
///
/// Replicate GPR rs value to all elements in vector (two signed 64-bit integer numbers).
/// If the source GPR is wider than the destination data format, the destination's elements
/// will be set to the least significant bits of the GPR.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fill.d))]
pub unsafe fn __msa_fill_d(a: i64) -> v2i64 {
    msa_fill_d(a)
}

/// Vector Floating-Point Base 2 Logarithm
///
/// The signed integral base 2 exponents of floating-point elements in vector `a`
/// (four 32-bit floating point numbers) are written as floating-point values to vector elements
/// (four 32-bit floating point numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(flog2.w))]
pub unsafe fn __msa_flog2_w(a: v4f32) -> v4f32 {
    msa_flog2_w(a)
}

/// Vector Floating-Point Base 2 Logarithm
///
/// The signed integral base 2 exponents of floating-point elements in vector `a`
/// (two 64-bit floating point numbers) are written as floating-point values to vector elements
/// (two 64-bit floating point numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(flog2.d))]
pub unsafe fn __msa_flog2_d(a: v2f64) -> v2f64 {
    msa_flog2_d(a)
}

/// Vector Floating-Point Multiply-Add
///
/// The floating-point elements in vector `b` (four 32-bit floating point numbers)
/// multiplied by floating-point elements in vector `c` (four 32-bit floating point numbers)
/// are added to the floating-point elements in vector `a` (four 32-bit floating point numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fmadd.w))]
pub unsafe fn __msa_fmadd_w(a: v4f32, b: v4f32, c: v4f32) -> v4f32 {
    msa_fmadd_w(a, mem::transmute(b), c)
}

/// Vector Floating-Point Multiply-Add
///
/// The floating-point elements in vector `b` (two 64-bit floating point numbers)
/// multiplied by floating-point elements in vector `c` (two 64-bit floating point numbers)
/// are added to the floating-point elements in vector `a` (two 64-bit floating point numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fmadd.d))]
pub unsafe fn __msa_fmadd_d(a: v2f64, b: v2f64, c: v2f64) -> v2f64 {
    msa_fmadd_d(a, mem::transmute(b), c)
}

/// Vector Floating-Point Maximum
///
/// The largest values between corresponding floating-point elements in vector `a`
/// (four 32-bit floating point numbers) and vector `b` (four 32-bit floating point numbers)
/// are written to vector (four 32-bit floating point numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fmax.w))]
pub unsafe fn __msa_fmax_w(a: v4f32, b: v4f32) -> v4f32 {
    msa_fmax_w(a, mem::transmute(b))
}

/// Vector Floating-Point Maximum
///
/// The largest values between corresponding floating-point elements in vector `a`
/// (two 64-bit floating point numbers) and vector `b` (two 64-bit floating point numbers)
/// are written to vector (two 64-bit floating point numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fmax.d))]
pub unsafe fn __msa_fmax_d(a: v2f64, b: v2f64) -> v2f64 {
    msa_fmax_d(a, mem::transmute(b))
}

/// Vector Floating-Point Maximum Based on Absolute Values
///
/// The value with the largest magnitude, i.e. absolute value, between corresponding
/// floating-point elements in vector `a` (four 32-bit floating point numbers)
/// and vector `b` (four 32-bit floating point numbers)
/// are written to vector (four 32-bit floating point numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fmax_a.w))]
pub unsafe fn __msa_fmax_a_w(a: v4f32, b: v4f32) -> v4f32 {
    msa_fmax_a_w(a, mem::transmute(b))
}

/// Vector Floating-Point Maximum Based on Absolute Values
///
/// The value with the largest magnitude, i.e. absolute value, between corresponding
/// floating-point elements in vector `a` (two 64-bit floating point numbers)
/// and vector `b` (two 64-bit floating point numbers)
/// are written to vector (two 64-bit floating point numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fmax_a.d))]
pub unsafe fn __msa_fmax_a_d(a: v2f64, b: v2f64) -> v2f64 {
    msa_fmax_a_d(a, mem::transmute(b))
}

/// Vector Floating-Point Minimum
///
/// The smallest values between corresponding floating-point elements in vector `a`
/// (four 32-bit floating point numbers) and vector `b` (four 32-bit floating point numbers)
/// are written to vector (four 32-bit floating point numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fmin.w))]
pub unsafe fn __msa_fmin_w(a: v4f32, b: v4f32) -> v4f32 {
    msa_fmin_w(a, mem::transmute(b))
}

/// Vector Floating-Point Minimum
///
/// The smallest values between corresponding floating-point elements in vector `a`
/// (two 64-bit floating point numbers) and vector `b` (two 64-bit floating point numbers)
/// are written to vector (two 64-bit floating point numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fmin.d))]
pub unsafe fn __msa_fmin_d(a: v2f64, b: v2f64) -> v2f64 {
    msa_fmin_d(a, mem::transmute(b))
}

/// Vector Floating-Point Minimum Based on Absolute Values
///
/// The value with the smallest magnitude, i.e. absolute value, between corresponding
/// floating-point elements in vector `a` (four 32-bit floating point numbers)
/// and vector `b` (four 32-bit floating point numbers)
/// are written to vector (four 32-bit floating point numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fmin_a.w))]
pub unsafe fn __msa_fmin_a_w(a: v4f32, b: v4f32) -> v4f32 {
    msa_fmin_a_w(a, mem::transmute(b))
}

/// Vector Floating-Point Minimum Based on Absolute Values
///
/// The value with the smallest magnitude, i.e. absolute value, between corresponding
/// floating-point elements in vector `a` (two 64-bit floating point numbers)
/// and vector `b` (two 64-bit floating point numbers)
/// are written to vector (two 64-bit floating point numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fmin_a.d))]
pub unsafe fn __msa_fmin_a_d(a: v2f64, b: v2f64) -> v2f64 {
    msa_fmin_a_d(a, mem::transmute(b))
}

/// Vector Floating-Point Multiply-Sub
///
/// The floating-point elements in vector `b` (four 32-bit floating point numbers)
/// multiplied by floating-point elements in vector `c` (four 32-bit floating point numbers)
/// are subtracted from the floating-point elements in vector `a` (four 32-bit floating point numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fmsub.w))]
pub unsafe fn __msa_fmsub_w(a: v4f32, b: v4f32, c: v4f32) -> v4f32 {
    msa_fmsub_w(a, mem::transmute(b), c)
}

/// Vector Floating-Point Multiply-Sub
///
/// The floating-point elements in vector `b` (two 64-bit floating point numbers)
/// multiplied by floating-point elements in vector `c` (two 64-bit floating point numbers)
/// are subtracted from the floating-point elements in vector `a` (two 64-bit floating point numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fmsub.d))]
pub unsafe fn __msa_fmsub_d(a: v2f64, b: v2f64, c: v2f64) -> v2f64 {
    msa_fmsub_d(a, mem::transmute(b), c)
}

/// Vector Floating-Point Multiplication
///
/// The floating-point elements in vector `a` (four 32-bit floating point numbers) are
/// multiplied by floating-point elements in vector `b` (four 32-bit floating point numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fmul.w))]
pub unsafe fn __msa_fmul_w(a: v4f32, b: v4f32) -> v4f32 {
    msa_fmul_w(a, mem::transmute(b))
}

/// Vector Floating-Point Multiplication
///
/// The floating-point elements in vector `a` (two 64-bit floating point numbers) are
/// multiplied by floating-point elements in vector `b` (two 64-bit floating point numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fmul.d))]
pub unsafe fn __msa_fmul_d(a: v2f64, b: v2f64) -> v2f64 {
    msa_fmul_d(a, mem::transmute(b))
}

/// Vector Floating-Point Round to Integer
///
/// The floating-point elements in vector `a` (four 32-bit floating point numbers)
/// are rounded to an integral valued floating-point number in the same format based
/// on the rounding mode bits RM in MSA Control and Status Register MSACSR.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(frint.w))]
pub unsafe fn __msa_frint_w(a: v4f32) -> v4f32 {
    msa_frint_w(a)
}

/// Vector Floating-Point Round to Integer
///
/// The floating-point elements in vector `a` (two 64-bit floating point numbers)
/// are rounded to an integral valued floating-point number in the same format based
/// on the rounding mode bits RM in MSA Control and Status Register MSACSR.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(frint.d))]
pub unsafe fn __msa_frint_d(a: v2f64) -> v2f64 {
    msa_frint_d(a)
}

/// Vector Approximate Floating-Point Reciprocal
///
/// The reciprocals of floating-point elements in vector `a` (four 32-bit floating point numbers)
/// are calculated and the result is written to vector (four 32-bit floating point numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(frcp.w))]
pub unsafe fn __msa_frcp_w(a: v4f32) -> v4f32 {
    msa_frcp_w(a)
}

/// Vector Approximate Floating-Point Reciprocal
///
/// The reciprocals of floating-point elements in vector `a` (two 64-bit floating point numbers)
/// are calculated and the result is written to vector (two 64-bit floating point numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(frcp.d))]
pub unsafe fn __msa_frcp_d(a: v2f64) -> v2f64 {
    msa_frcp_d(a)
}

/// Vector Approximate Floating-Point Reciprocal of Square Root
///
/// The reciprocals of the square roots of floating-point elements in vector `a` (four 32-bit floating point numbers)
/// are calculated and the result is written to vector (four 32-bit floating point numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(frsqrt.w))]
pub unsafe fn __msa_frsqrt_w(a: v4f32) -> v4f32 {
    msa_frsqrt_w(a)
}

/// Vector Approximate Floating-Point Reciprocal of Square Root
///
/// The reciprocals of the square roots of floating-point elements in vector `a` (two 64-bit floating point numbers)
/// are calculated and the result is written to vector (two 64-bit floating point numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(frsqrt.d))]
pub unsafe fn __msa_frsqrt_d(a: v2f64) -> v2f64 {
    msa_frsqrt_d(a)
}

/// Vector Floating-Point Signaling Compare Always False
///
/// Set all bits to 0 in vector (four signed 32-bit integer numbers) elements.
/// Signaling and quiet NaN elements in vector `a` (four 32-bit floating point numbers)
/// or `b` (four 32-bit floating point numbers) signal Invalid Operation exception.
/// In case of a floating-point exception, the default result has all bits set to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fsaf.w))]
pub unsafe fn __msa_fsaf_w(a: v4f32, b: v4f32) -> v4i32 {
    msa_fsaf_w(a, mem::transmute(b))
}

/// Vector Floating-Point Signaling Compare Always False
///
/// Set all bits to 0 in vector (two signed 64-bit integer numbers) elements.
/// Signaling and quiet NaN elements in vector `a` (two 64-bit floating point numbers)
/// or `b` (two 64-bit floating point numbers) signal Invalid Operation exception.
/// In case of a floating-point exception, the default result has all bits set to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fsaf.d))]
pub unsafe fn __msa_fsaf_d(a: v2f64, b: v2f64) -> v2i64 {
    msa_fsaf_d(a, mem::transmute(b))
}

/// Vector Floating-Point Signaling Compare Equal
///
/// Set all bits to 1 in vector (four signed 32-bit integer numbers) elements
/// if the corresponding `a` (four 32-bit floating point numbers)
/// and `b` (four 32-bit floating point numbers) elements are equal, otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fseq.w))]
pub unsafe fn __msa_fseq_w(a: v4f32, b: v4f32) -> v4i32 {
    msa_fseq_w(a, mem::transmute(b))
}

/// Vector Floating-Point Signaling Compare Equal
///
/// Set all bits to 1 in vector (two signed 64-bit integer numbers) elements
/// if the corresponding `a` (two 64-bit floating point numbers)
/// and `b` (two 64-bit floating point numbers) elements are equal, otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fseq.d))]
pub unsafe fn __msa_fseq_d(a: v2f64, b: v2f64) -> v2i64 {
    msa_fseq_d(a, mem::transmute(b))
}

/// Vector Floating-Point Signaling Compare Less or Equal
///
/// Set all bits to 1 in vector (four signed 32-bit integer numbers) elements
/// if the corresponding `a` (four 32-bit floating point numbers) elements
/// are less than or equal to `b` (four 32-bit floating point numbers) elements, otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fsle.w))]
pub unsafe fn __msa_fsle_w(a: v4f32, b: v4f32) -> v4i32 {
    msa_fsle_w(a, mem::transmute(b))
}

/// Vector Floating-Point Signaling Compare Less or Equal
///
/// Set all bits to 1 in vector (two signed 64-bit integer numbers) elements
/// if the corresponding `a` (two 64-bit floating point numbers) elements
/// are less than or equal to `b` (two 64-bit floating point numbers) elements, otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fsle.d))]
pub unsafe fn __msa_fsle_d(a: v2f64, b: v2f64) -> v2i64 {
    msa_fsle_d(a, mem::transmute(b))
}

/// Vector Floating-Point Signaling Compare Less Than
///
/// Set all bits to 1 in vector (four signed 32-bit integer numbers) elements
/// if the corresponding `a` (four 32-bit floating point numbers) elements
/// are less than `b` (four 32-bit floating point numbers) elements, otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fslt.w))]
pub unsafe fn __msa_fslt_w(a: v4f32, b: v4f32) -> v4i32 {
    msa_fslt_w(a, mem::transmute(b))
}

/// Vector Floating-Point Signaling Compare Less Than
///
/// Set all bits to 1 in vector (two signed 64-bit integer numbers) elements
/// if the corresponding `a` (two 64-bit floating point numbers) elements
/// are less than `b` (two 64-bit floating point numbers) elements, otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fslt.d))]
pub unsafe fn __msa_fslt_d(a: v2f64, b: v2f64) -> v2i64 {
    msa_fslt_d(a, mem::transmute(b))
}

/// Vector Floating-Point Signaling Compare Not Equal
///
/// Set all bits to 1 in vector (four signed 32-bit integer numbers) elements
/// if the corresponding `a` (four 32-bit floating point numbers) and
/// `b` (four 32-bit floating point numbers) elements are not equal, otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fsne.w))]
pub unsafe fn __msa_fsne_w(a: v4f32, b: v4f32) -> v4i32 {
    msa_fsne_w(a, mem::transmute(b))
}

/// Vector Floating-Point Signaling Compare Not Equal
///
/// Set all bits to 1 in vector (two signed 64-bit integer numbers) elements
/// if the corresponding `a` (two 64-bit floating point numbers) and
/// `b` (two 64-bit floating point numbers) elements are not equal, otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fsne.d))]
pub unsafe fn __msa_fsne_d(a: v2f64, b: v2f64) -> v2i64 {
    msa_fsne_d(a, mem::transmute(b))
}

/// Vector Floating-Point Signaling Compare Ordered
///
/// Set all bits to 1 in vector (four signed 32-bit integer numbers) elements
/// if the corresponding `a` (four 32-bit floating point numbers) and
/// `b` (four 32-bit floating point numbers) elements are ordered,
/// i.e. both elements are not NaN values, otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fsor.w))]
pub unsafe fn __msa_fsor_w(a: v4f32, b: v4f32) -> v4i32 {
    msa_fsor_w(a, mem::transmute(b))
}

/// Vector Floating-Point Signaling Compare Ordered
///
/// Set all bits to 1 in vector (two signed 64-bit integer numbers) elements
/// if the corresponding `a` (two 64-bit floating point numbers) and
/// `b` (two 64-bit floating point numbers) elements are ordered,
/// i.e. both elements are not NaN values, otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fsor.d))]
pub unsafe fn __msa_fsor_d(a: v2f64, b: v2f64) -> v2i64 {
    msa_fsor_d(a, mem::transmute(b))
}

/// Vector Floating-Point Square Root
///
/// The square roots of floating-point elements in vector `a`
/// (four 32-bit floating point numbers) are written to vector
/// (four 32-bit floating point numbers) elements are ordered,.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fsqrt.w))]
pub unsafe fn __msa_fsqrt_w(a: v4f32) -> v4f32 {
    msa_fsqrt_w(a)
}

/// Vector Floating-Point Square Root
///
/// The square roots of floating-point elements in vector `a`
/// (two 64-bit floating point numbers) are written to vector
/// (two 64-bit floating point numbers) elements are ordered,.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fsqrt.d))]
pub unsafe fn __msa_fsqrt_d(a: v2f64) -> v2f64 {
    msa_fsqrt_d(a)
}

/// Vector Floating-Point Subtraction
///
/// The floating-point elements in vector `b` (four 32-bit floating point numbers)
/// are subtracted from the floating-point elements in vector `a`
/// (four 32-bit floating point numbers).
/// The result is written to vector (four 32-bit floating point numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fsub.w))]
pub unsafe fn __msa_fsub_w(a: v4f32, b: v4f32) -> v4f32 {
    msa_fsub_w(a, mem::transmute(b))
}

/// Vector Floating-Point Subtraction
///
/// The floating-point elements in vector `b` (two 64-bit floating point numbers)
/// are subtracted from the floating-point elements in vector `a`
/// (two 64-bit floating point numbers).
/// The result is written to vector (two 64-bit floating point numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fsub.d))]
pub unsafe fn __msa_fsub_d(a: v2f64, b: v2f64) -> v2f64 {
    msa_fsub_d(a, mem::transmute(b))
}

/// Vector Floating-Point Signaling Compare Ordered
///
/// Set all bits to 1 in vector (four signed 32-bit integer numbers) elements
/// if the corresponding `a` (four 32-bit floating point numbers) and
/// `b` (four 32-bit floating point numbers) elements are unordered or equal,
/// otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fsueq.w))]
pub unsafe fn __msa_fsueq_w(a: v4f32, b: v4f32) -> v4i32 {
    msa_fsueq_w(a, mem::transmute(b))
}

/// Vector Floating-Point Signaling Compare Ordered
///
/// Set all bits to 1 in vector (two signed 64-bit integer numbers) elements
/// if the corresponding `a` (two 64-bit floating point numbers) and
/// `b` (two 64-bit floating point numbers) elements are unordered or equal,
/// otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fsueq.d))]
pub unsafe fn __msa_fsueq_d(a: v2f64, b: v2f64) -> v2i64 {
    msa_fsueq_d(a, mem::transmute(b))
}

/// Vector Floating-Point Signaling Compare Unordered or Less or Equal
///
/// Set all bits to 1 in vector (four signed 32-bit integer numbers) elements
/// if the corresponding `a` (four 32-bit floating point numbers) elements are
/// unordered or less than or equal to `b` (four 32-bit floating point numbers) elements,
/// otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fsule.w))]
pub unsafe fn __msa_fsule_w(a: v4f32, b: v4f32) -> v4i32 {
    msa_fsule_w(a, mem::transmute(b))
}

/// Vector Floating-Point Signaling Compare Unordered or Less or Equal
///
/// Set all bits to 1 in vector (two signed 64-bit integer numbers) elements
/// if the corresponding `a` (two 64-bit floating point numbers) elements are
/// unordered or less than or equal to `b` (two 64-bit floating point numbers) elements,
/// otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fsule.d))]
pub unsafe fn __msa_fsule_d(a: v2f64, b: v2f64) -> v2i64 {
    msa_fsule_d(a, mem::transmute(b))
}

/// Vector Floating-Point Signaling Compare Unordered or Less Than
///
/// Set all bits to 1 in vector (four signed 32-bit integer numbers) elements
/// if the corresponding `a` (four 32-bit floating point numbers) elements
/// are unordered or less than `b` (four 32-bit floating point numbers) elements,
/// otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fsult.w))]
pub unsafe fn __msa_fsult_w(a: v4f32, b: v4f32) -> v4i32 {
    msa_fsult_w(a, mem::transmute(b))
}

/// Vector Floating-Point Signaling Compare Unordered or Less Than
///
/// Set all bits to 1 in vector (two signed 64-bit integer numbers) elements
/// if the corresponding `a` (two 64-bit floating point numbers) elements
/// are unordered or less than `b` (two 64-bit floating point numbers) elements,
/// otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fsult.d))]
pub unsafe fn __msa_fsult_d(a: v2f64, b: v2f64) -> v2i64 {
    msa_fsult_d(a, mem::transmute(b))
}

/// Vector Floating-Point Signaling Compare Unordered
///
/// Set all bits to 1 in vector (four signed 32-bit integer numbers) elements
/// if the corresponding `a` (four 32-bit floating point numbers) and
/// `b` (four 32-bit floating point numbers) elements are unordered,
/// i.e. at least one element is a NaN value, otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fsun.w))]
pub unsafe fn __msa_fsun_w(a: v4f32, b: v4f32) -> v4i32 {
    msa_fsun_w(a, mem::transmute(b))
}

/// Vector Floating-Point Signaling Compare Unordered
///
/// Set all bits to 1 in vector (two signed 64-bit integer numbers) elements
/// if the corresponding `a` (two 64-bit floating point numbers) and
/// `b` (two 64-bit floating point numbers) elements are unordered,
/// i.e. at least one element is a NaN value, otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fsun.d))]
pub unsafe fn __msa_fsun_d(a: v2f64, b: v2f64) -> v2i64 {
    msa_fsun_d(a, mem::transmute(b))
}

/// Vector Floating-Point Signaling Compare Unordered or Not Equal
///
/// Set all bits to 1 in vector (four signed 32-bit integer numbers) elements
/// if the corresponding `a` (four 32-bit floating point numbers) and
/// `b` (four 32-bit floating point numbers) elements are unordered or not equal,
/// otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fsune.w))]
pub unsafe fn __msa_fsune_w(a: v4f32, b: v4f32) -> v4i32 {
    msa_fsune_w(a, mem::transmute(b))
}

/// Vector Floating-Point Signaling Compare Unordered or Not Equal
///
/// Set all bits to 1 in vector (two signed 64-bit integer numbers) elements
/// if the corresponding `a` (two 64-bit floating point numbers) and
/// `b` (two 64-bit floating point numbers) elements are unordered or not equal,
/// otherwise set all bits to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(fsune.d))]
pub unsafe fn __msa_fsune_d(a: v2f64, b: v2f64) -> v2i64 {
    msa_fsune_d(a, mem::transmute(b))
}

/// Vector Floating-Point Convert to Signed Integer
///
///The elements in vector `a` (four 32-bit floating point numbers)
/// are rounded and converted to signed integer values based on the
/// rounding mode bits RM in MSA Control and Status Register MSACSR.
/// The result is written to vector (four signed 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ftint_s.w))]
pub unsafe fn __msa_ftint_s_w(a: v4f32) -> v4i32 {
    msa_ftint_s_w(a)
}

/// Vector Floating-Point Convert to Signed Integer
///
///The elements in vector `a` (two 64-bit floating point numbers)
/// are rounded and converted to signed integer values based on the
/// rounding mode bits RM in MSA Control and Status Register MSACSR.
/// The result is written to vector (two signed 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ftint_s.d))]
pub unsafe fn __msa_ftint_s_d(a: v2f64) -> v2i64 {
    msa_ftint_s_d(a)
}

/// Vector Floating-Point Convert to Unsigned Integer
///
/// The elements in vector `a` (four 32-bit floating point numbers)
/// are rounded and converted to signed integer values based on the
/// rounding mode bits RM in MSA Control and Status Register MSACSR.
/// The result is written to vector (four unsigned 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ftint_u.w))]
pub unsafe fn __msa_ftint_u_w(a: v4f32) -> v4u32 {
    msa_ftint_u_w(a)
}

/// Vector Floating-Point Convert to Unsigned Integer
///
/// The elements in vector `a` (two 64-bit floating point numbers)
/// are rounded and converted to signed integer values based on the
/// rounding mode bits RM in MSA Control and Status Register MSACSR.
/// The result is written to vector (two unsigned 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ftint_u.d))]
pub unsafe fn __msa_ftint_u_d(a: v2f64) -> v2u64 {
    msa_ftint_u_d(a)
}

/// Vector Floating-Point Convert to Fixed-Point
///
/// The elements in vector `a` (four 32-bit floating point numbers)
/// and `b` (four 32-bit floating point numbers) are down-converted to a fixed-point
/// representation, i.e. from 64-bit floating-point to 32-bit Q31 fixed-point
/// representation, or from 32-bit floating-point to 16-bit Q15 fixed-point representation.
/// The result is written to vector (eight signed 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ftq.h))]
pub unsafe fn __msa_ftq_h(a: v4f32, b: v4f32) -> v8i16 {
    msa_ftq_h(a, mem::transmute(b))
}

/// Vector Floating-Point Convert to Fixed-Point
///
/// The elements in vector `a` (two 64-bit floating point numbers)
/// and `b` (two 64-bit floating point numbers) are down-converted to a fixed-point
/// representation, i.e. from 64-bit floating-point to 32-bit Q31 fixed-point
/// representation, or from 32-bit floating-point to 16-bit Q15 fixed-point representation.
/// The result is written to vector (four signed 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ftq.w))]
pub unsafe fn __msa_ftq_w(a: v2f64, b: v2f64) -> v4i32 {
    msa_ftq_w(a, mem::transmute(b))
}

/// Vector Floating-Point Truncate and Convert to Signed Integer
///
/// The elements in vector `a` (four 32-bit floating point numbers)
/// are truncated, i.e. rounded toward zero, to signed integer values.
/// The result is written to vector (four signed 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ftrunc_s.w))]
pub unsafe fn __msa_ftrunc_s_w(a: v4f32) -> v4i32 {
    msa_ftrunc_s_w(a)
}

/// Vector Floating-Point Truncate and Convert to Signed Integer
///
/// The elements in vector `a` (two 64-bit floating point numbers)
/// are truncated, i.e. rounded toward zero, to signed integer values.
/// The result is written to vector (two signed 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ftrunc_s.d))]
pub unsafe fn __msa_ftrunc_s_d(a: v2f64) -> v2i64 {
    msa_ftrunc_s_d(a)
}

/// Vector Floating-Point Truncate and Convert to Unsigned Integer
///
/// The elements in vector `a` (four 32-bit floating point numbers)
/// are truncated, i.e. rounded toward zero, to unsigned integer values.
/// The result is written to vector (four unsigned 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ftrunc_u.w))]
pub unsafe fn __msa_ftrunc_u_w(a: v4f32) -> v4u32 {
    msa_ftrunc_u_w(a)
}

/// Vector Floating-Point Truncate and Convert to Unsigned Integer
///
/// The elements in vector `a` (two 64-bit floating point numbers)
/// are truncated, i.e. rounded toward zero, to unsigned integer values.
/// The result is written to vector (two unsigned 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ftrunc_u.d))]
pub unsafe fn __msa_ftrunc_u_d(a: v2f64) -> v2u64 {
    msa_ftrunc_u_d(a)
}

/// Vector Signed Horizontal Add
///
/// The sign-extended odd elements in vector `a` (sixteen signed 8-bit integer numbers)
/// are added to the sign-extended even elements in vector `b` (sixteen signed 8-bit integer numbers)
/// producing a result twice the size of the input operands.
/// The result is written to vector (eight signed 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(hadd_s.h))]
pub unsafe fn __msa_hadd_s_h(a: v16i8, b: v16i8) -> v8i16 {
    msa_hadd_s_h(a, mem::transmute(b))
}

/// Vector Signed Horizontal Add
///
/// The sign-extended odd elements in vector `a` (eight signed 16-bit integer numbers)
/// are added to the sign-extended even elements in vector `b` (eight signed 16-bit integer numbers)
/// producing a result twice the size of the input operands.
/// The result is written to vector (four signed 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(hadd_s.w))]
pub unsafe fn __msa_hadd_s_w(a: v8i16, b: v8i16) -> v4i32 {
    msa_hadd_s_w(a, mem::transmute(b))
}

/// Vector Signed Horizontal Add
///
/// The sign-extended odd elements in vector `a` (four signed 32-bit integer numbers)
/// are added to the sign-extended even elements in vector `b` (four signed 32-bit integer numbers)
/// producing a result twice the size of the input operands.
/// The result is written to vector (two signed 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(hadd_s.d))]
pub unsafe fn __msa_hadd_s_d(a: v4i32, b: v4i32) -> v2i64 {
    msa_hadd_s_d(a, mem::transmute(b))
}

/// Vector Unsigned Horizontal Add
///
/// The zero-extended odd elements in vector `a` (sixteen unsigned 8-bit integer numbers)
/// are added to the zero-extended even elements in vector `b` (sixteen unsigned 8-bit integer numbers)
/// producing a result twice the size of the input operands.
/// The result is written to vector (eight unsigned 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(hadd_u.h))]
pub unsafe fn __msa_hadd_u_h(a: v16u8, b: v16u8) -> v8u16 {
    msa_hadd_u_h(a, mem::transmute(b))
}

/// Vector Unsigned Horizontal Add
///
/// The zero-extended odd elements in vector `a` (eight unsigned 16-bit integer numbers)
/// are added to the zero-extended even elements in vector `b` (eight unsigned 16-bit integer numbers)
/// producing a result twice the size of the input operands.
/// The result is written to vector (four unsigned 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(hadd_u.w))]
pub unsafe fn __msa_hadd_u_w(a: v8u16, b: v8u16) -> v4u32 {
    msa_hadd_u_w(a, mem::transmute(b))
}

/// Vector Unsigned Horizontal Add
///
/// The zero-extended odd elements in vector `a` (four unsigned 32-bit integer numbers)
/// are added to the zero-extended even elements in vector `b` (four unsigned 32-bit integer numbers)
/// producing a result twice the size of the input operands.
/// The result is written to vector (two unsigned 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(hadd_u.d))]
pub unsafe fn __msa_hadd_u_d(a: v4u32, b: v4u32) -> v2u64 {
    msa_hadd_u_d(a, mem::transmute(b))
}

/// Vector Signed Horizontal Subtract
///
/// The sign-extended odd elements in vector `b` (sixteen signed 8-bit integer numbers)
/// are subtracted from the sign-extended elements in vector `a` (sixteen signed 8-bit integer numbers)
/// producing a result twice the size of the input operands.
/// The result is written to vector (eight signed 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(hsub_s.h))]
pub unsafe fn __msa_hsub_s_h(a: v16i8, b: v16i8) -> v8i16 {
    msa_hsub_s_h(a, mem::transmute(b))
}

/// Vector Signed Horizontal Subtract
///
/// The sign-extended odd elements in vector `b` (eight signed 16-bit integer numbers)
/// are subtracted from the sign-extended elements in vector `a` (eight signed 16-bit integer numbers)
/// producing a result twice the size of the input operands.
/// The result is written to vector (four signed 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(hsub_s.w))]
pub unsafe fn __msa_hsub_s_w(a: v8i16, b: v8i16) -> v4i32 {
    msa_hsub_s_w(a, mem::transmute(b))
}

/// Vector Signed Horizontal Subtract
///
/// The sign-extended odd elements in vector `b` (four signed 32-bit integer numbers)
/// are subtracted from the sign-extended elements in vector `a` (four signed 32-bit integer numbers)
/// producing a result twice the size of the input operands.
/// The result is written to vector (two signed 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(hsub_s.d))]
pub unsafe fn __msa_hsub_s_d(a: v4i32, b: v4i32) -> v2i64 {
    msa_hsub_s_d(a, mem::transmute(b))
}

/// Vector Unsigned Horizontal Subtract
///
/// The zero-extended odd elements in vector `b` (sixteen unsigned 8-bit integer numbers)
/// are subtracted from the zero-extended elements in vector `a` (sixteen unsigned 8-bit integer numbers)
/// producing a result twice the size of the input operands.
/// The result is written to vector (eight signed 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(hsub_u.h))]
pub unsafe fn __msa_hsub_u_h(a: v16u8, b: v16u8) -> v8i16 {
    msa_hsub_u_h(a, mem::transmute(b))
}

/// Vector Unsigned Horizontal Subtract
///
/// The zero-extended odd elements in vector `b` (eight unsigned 16-bit integer numbers)
/// are subtracted from the zero-extended elements in vector `a` (eight unsigned 16-bit integer numbers)
/// producing a result twice the size of the input operands.
/// The result is written to vector (four signed 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(hsub_u.w))]
pub unsafe fn __msa_hsub_u_w(a: v8u16, b: v8u16) -> v4i32 {
    msa_hsub_u_w(a, mem::transmute(b))
}

/// Vector Unsigned Horizontal Subtract
///
/// The zero-extended odd elements in vector `b` (four unsigned 32-bit integer numbers)
/// are subtracted from the zero-extended elements in vector `a` (four unsigned 32-bit integer numbers)
/// producing a result twice the size of the input operands.
/// The result is written to vector (two signed 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(hsub_u.d))]
pub unsafe fn __msa_hsub_u_d(a: v4u32, b: v4u32) -> v2i64 {
    msa_hsub_u_d(a, mem::transmute(b))
}

/// Vector Interleave Even
///
/// Even elements in vectors `a` (sixteen signed 8-bit integer numbers)
/// and vector `b` (sixteen signed 8-bit integer numbers) are copied to the result
/// (sixteen signed 8-bit integer numbers)
/// alternating one element from `a` with one element from `b`.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ilvev.b))]
pub unsafe fn __msa_ilvev_b(a: v16i8, b: v16i8) -> v16i8 {
    msa_ilvev_b(a, mem::transmute(b))
}

/// Vector Interleave Even
///
/// Even elements in vectors `a` (eight signed 16-bit integer numbers)
/// and vector `b` (eight signed 16-bit integer numbers) are copied to the result
/// (eight signed 16-bit integer numbers)
/// alternating one element from `a` with one element from `b`.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ilvev.h))]
pub unsafe fn __msa_ilvev_h(a: v8i16, b: v8i16) -> v8i16 {
    msa_ilvev_h(a, mem::transmute(b))
}

/// Vector Interleave Even
///
/// Even elements in vectors `a` (four signed 32-bit integer numbers)
/// and vector `b` (four signed 32-bit integer numbers) are copied to the result
/// (four signed 32-bit integer numbers)
/// alternating one element from `a` with one element from `b`.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ilvev.w))]
pub unsafe fn __msa_ilvev_w(a: v4i32, b: v4i32) -> v4i32 {
    msa_ilvev_w(a, mem::transmute(b))
}

/// Vector Interleave Even
///
/// Even elements in vectors `a` (two signed 64-bit integer numbers)
/// and vector `b` (two signed 64-bit integer numbers) are copied to the result
/// (two signed 64-bit integer numbers)
/// alternating one element from `a` with one element from `b`.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ilvev.d))]
pub unsafe fn __msa_ilvev_d(a: v2i64, b: v2i64) -> v2i64 {
    msa_ilvev_d(a, mem::transmute(b))
}

/// Vector Interleave Left
///
/// The left half elements in vectors `a` (sixteen signed 8-bit integer numbers)
/// and vector `b` (sixteen signed 8-bit integer numbers) are copied to the result
/// (sixteen signed 8-bit integer numbers)
/// alternating one element from `a` with one element from `b`.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ilvl.b))]
pub unsafe fn __msa_ilvl_b(a: v16i8, b: v16i8) -> v16i8 {
    msa_ilvl_b(a, mem::transmute(b))
}

/// Vector Interleave Left
///
/// The left half elements in vectors `a` (eight signed 16-bit integer numbers)
/// and vector `b` (eight signed 16-bit integer numbers) are copied to the result
/// (eight signed 16-bit integer numbers)
/// alternating one element from `a` with one element from `b`.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ilvl.h))]
pub unsafe fn __msa_ilvl_h(a: v8i16, b: v8i16) -> v8i16 {
    msa_ilvl_h(a, mem::transmute(b))
}

/// Vector Interleave Left
///
/// The left half elements in vectors `a` (four signed 32-bit integer numbers)
/// and vector `b` (four signed 32-bit integer numbers) are copied to the result
/// (four signed 32-bit integer numbers)
/// alternating one element from `a` with one element from `b`.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ilvl.w))]
pub unsafe fn __msa_ilvl_w(a: v4i32, b: v4i32) -> v4i32 {
    msa_ilvl_w(a, mem::transmute(b))
}

/// Vector Interleave Left
///
/// The left half elements in vectors `a` (two signed 64-bit integer numbers)
/// and vector `b` (two signed 64-bit integer numbers) are copied to the result
/// (two signed 64-bit integer numbers)
/// alternating one element from `a` with one element from `b`.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ilvl.d))]
pub unsafe fn __msa_ilvl_d(a: v2i64, b: v2i64) -> v2i64 {
    msa_ilvl_d(a, mem::transmute(b))
}

/// Vector Interleave Odd
///
/// Odd elements in vectors `a` (sixteen signed 8-bit integer numbers)
/// and vector `b` (sixteen signed 8-bit integer numbers) are copied to the result
/// (sixteen signed 8-bit integer numbers)
/// alternating one element from `a` with one element from `b`.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ilvod.b))]
pub unsafe fn __msa_ilvod_b(a: v16i8, b: v16i8) -> v16i8 {
    msa_ilvod_b(a, mem::transmute(b))
}

/// Vector Interleave Odd
///
/// Odd elements in vectors `a` (eight signed 16-bit integer numbers)
/// and vector `b` (eight signed 16-bit integer numbers) are copied to the result
/// (eight signed 16-bit integer numbers)
/// alternating one element from `a` with one element from `b`.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ilvod.h))]
pub unsafe fn __msa_ilvod_h(a: v8i16, b: v8i16) -> v8i16 {
    msa_ilvod_h(a, mem::transmute(b))
}

/// Vector Interleave Odd
///
/// Odd elements in vectors `a` (four signed 32-bit integer numbers)
/// and vector `b` (four signed 32-bit integer numbers) are copied to the result
/// (four signed 32-bit integer numbers)
/// alternating one element from `a` with one element from `b`.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ilvod.w))]
pub unsafe fn __msa_ilvod_w(a: v4i32, b: v4i32) -> v4i32 {
    msa_ilvod_w(a, mem::transmute(b))
}

/// Vector Interleave Odd
///
/// Odd elements in vectors `a` (two signed 64-bit integer numbers)
/// and vector `b` (two signed 64-bit integer numbers) are copied to the result
/// (two signed 64-bit integer numbers)
/// alternating one element from `a` with one element from `b`.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ilvod.d))]
pub unsafe fn __msa_ilvod_d(a: v2i64, b: v2i64) -> v2i64 {
    msa_ilvod_d(a, mem::transmute(b))
}

/// Vector Interleave Right
///
/// The right half elements in vectors `a` (sixteen signed 8-bit integer numbers)
/// and vector `b` (sixteen signed 8-bit integer numbers) are copied to the result
/// (sixteen signed 8-bit integer numbers)
/// alternating one element from `a` with one element from `b`.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ilvr.b))]
pub unsafe fn __msa_ilvr_b(a: v16i8, b: v16i8) -> v16i8 {
    msa_ilvr_b(a, mem::transmute(b))
}

/// Vector Interleave Right
///
/// The right half elements in vectors `a` (eight signed 16-bit integer numbers)
/// and vector `b` (eight signed 16-bit integer numbers) are copied to the result
/// (eight signed 16-bit integer numbers)
/// alternating one element from `a` with one element from `b`.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ilvr.h))]
pub unsafe fn __msa_ilvr_h(a: v8i16, b: v8i16) -> v8i16 {
    msa_ilvr_h(a, mem::transmute(b))
}

/// Vector Interleave Right
///
/// The right half elements in vectors `a` (four signed 32-bit integer numbers)
/// and vector `b` (four signed 32-bit integer numbers) are copied to the result
/// (four signed 32-bit integer numbers)
/// alternating one element from `a` with one element from `b`.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ilvr.w))]
pub unsafe fn __msa_ilvr_w(a: v4i32, b: v4i32) -> v4i32 {
    msa_ilvr_w(a, mem::transmute(b))
}

/// Vector Interleave Right
///
/// The right half elements in vectors `a` (two signed 64-bit integer numbers)
/// and vector `b` (two signed 64-bit integer numbers) are copied to the result
/// (two signed 64-bit integer numbers)
/// alternating one element from `a` with one element from `b`.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ilvr.d))]
pub unsafe fn __msa_ilvr_d(a: v2i64, b: v2i64) -> v2i64 {
    msa_ilvr_d(a, mem::transmute(b))
}

/// GPR Insert Element
///
/// Set element `imm4` in vector `a` (sixteen signed 8-bit integer numbers) to GPR `c` value.
/// All other elements in vector `a` are unchanged. If the source GPR is wider than the
/// destination data format, the destination's elements will be set to the least significant bits of the GPR.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(insert.b, imm4 = 0b1111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_insert_b<const IMM4: i32>(a: v16i8, c: i32) -> v16i8 {
    static_assert_imm4!(IMM4);
    msa_insert_b(a, IMM4, c)
}

/// GPR Insert Element
///
/// Set element `imm3` in vector `a` (eight signed 16-bit integer numbers) to GPR `c` value.
/// All other elements in vector `a` are unchanged. If the source GPR is wider than the
/// destination data format, the destination's elements will be set to the least significant bits of the GPR.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(insert.h, imm3 = 0b111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_insert_h<const IMM3: i32>(a: v8i16, c: i32) -> v8i16 {
    static_assert_imm3!(IMM3);
    msa_insert_h(a, IMM3, c)
}

/// GPR Insert Element
///
/// Set element `imm2` in vector `a` (four signed 32-bit integer numbers) to GPR `c` value.
/// All other elements in vector `a` are unchanged. If the source GPR is wider than the
/// destination data format, the destination's elements will be set to the least significant bits of the GPR.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(insert.w, imm2 = 0b11))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_insert_w<const IMM2: i32>(a: v4i32, c: i32) -> v4i32 {
    static_assert_imm2!(IMM2);
    msa_insert_w(a, IMM2, c)
}

/// GPR Insert Element
///
/// Set element `imm1` in vector `a` (two signed 64-bit integer numbers) to GPR `c` value.
/// All other elements in vector `a` are unchanged. If the source GPR is wider than the
/// destination data format, the destination's elements will be set to the least significant bits of the GPR.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(insert.d, imm1 = 0b1))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_insert_d<const IMM1: i32>(a: v2i64, c: i64) -> v2i64 {
    static_assert_imm1!(IMM1);
    msa_insert_d(a, IMM1, c)
}

/// Element Insert Element
///
/// Set element `imm1` in the result vector `a` (sixteen signed 8-bit integer numbers) to element 0
/// in vector `c` (sixteen signed 8-bit integer numbers) value.
/// All other elements in vector `a` are unchanged.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(insve.b, imm4 = 0b1111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_insve_b<const IMM4: i32>(a: v16i8, c: v16i8) -> v16i8 {
    static_assert_imm4!(IMM4);
    msa_insve_b(a, IMM4, c)
}

/// Element Insert Element
///
/// Set element `imm1` in the result vector `a` (eight signed 16-bit integer numbers) to element 0
/// in vector `c` (eight signed 16-bit integer numbers) value.
/// All other elements in vector `a` are unchanged.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(insve.h, imm3 = 0b111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_insve_h<const IMM3: i32>(a: v8i16, c: v8i16) -> v8i16 {
    static_assert_imm3!(IMM3);
    msa_insve_h(a, IMM3, c)
}

/// Element Insert Element
///
/// Set element `imm1` in the result vector `a` (four signed 32-bit integer numbers) to element 0
/// in vector `c` (four signed 32-bit integer numbers) value.
/// All other elements in vector `a` are unchanged.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(insve.w, imm2 = 0b11))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_insve_w<const IMM2: i32>(a: v4i32, c: v4i32) -> v4i32 {
    static_assert_imm2!(IMM2);
    msa_insve_w(a, IMM2, c)
}

/// Element Insert Element
///
/// Set element `imm1` in the result vector `a` (two signed 64-bit integer numbers) to element 0
/// in vector `c` (two signed 64-bit integer numbers) value.
/// All other elements in vector `a` are unchanged.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(insve.d, imm1 = 0b1))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_insve_d<const IMM1: i32>(a: v2i64, c: v2i64) -> v2i64 {
    static_assert_imm1!(IMM1);
    msa_insve_d(a, IMM1, c)
}

/// Vector Load
///
/// The WRLEN / 8 bytes at the effective memory location addressed by the base
/// `mem_addr` and the 10-bit signed immediate offset `imm_s10` are fetched and placed in
/// the vector (sixteen signed 8-bit integer numbers) value.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ld.b, imm_s10 = 0b1111111111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_ld_b<const IMM_S10: i32>(mem_addr: *mut u8) -> v16i8 {
    static_assert_imm_s10!(IMM_S10);
    msa_ld_b(mem_addr, IMM_S10)
}

/// Vector Load
///
/// The WRLEN / 8 bytes at the effective memory location addressed by the base
/// `mem_addr` and the 10-bit signed immediate offset `imm_s11` are fetched and placed in
/// the vector (eight signed 16-bit integer numbers) value.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ld.h, imm_s11 = 0b11111111111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_ld_h<const IMM_S11: i32>(mem_addr: *mut u8) -> v8i16 {
    static_assert_imm_s11!(IMM_S11);
    static_assert!(IMM_S11: i32 where IMM_S11 % 2 == 0);
    msa_ld_h(mem_addr, IMM_S11)
}

/// Vector Load
///
/// The WRLEN / 8 bytes at the effective memory location addressed by the base
/// `mem_addr` and the 10-bit signed immediate offset `imm_s12` are fetched and placed in
/// the vector (four signed 32-bit integer numbers) value.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ld.w, imm_s12 = 0b111111111111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_ld_w<const IMM_S12: i32>(mem_addr: *mut u8) -> v4i32 {
    static_assert_imm_s12!(IMM_S12);
    static_assert!(IMM_S12: i32 where IMM_S12 % 4 == 0);
    msa_ld_w(mem_addr, IMM_S12)
}

/// Vector Load
///
/// The WRLEN / 8 bytes at the effective memory location addressed by the base
/// `mem_addr` and the 10-bit signed immediate offset `imm_s13` are fetched and placed in
/// the vector (two signed 64-bit integer numbers) value.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ld.d, imm_s13 = 0b1111111111111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_ld_d<const IMM_S13: i32>(mem_addr: *mut u8) -> v2i64 {
    static_assert_imm_s13!(IMM_S13);
    static_assert!(IMM_S13: i32 where IMM_S13 % 8 == 0);
    msa_ld_d(mem_addr, IMM_S13)
}

/// Immediate Load
///
/// The signed immediate imm_s10 is replicated in all vector
/// (sixteen signed 8-bit integer numbers) elements. For byte elements,
/// only the least significant 8 bits of imm_s10 will be used.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ldi.b, imm_s10 = 0b1111111111))]
#[rustc_legacy_const_generics(0)]
pub unsafe fn __msa_ldi_b<const IMM_S10: i32>() -> v16i8 {
    static_assert_imm_s10!(IMM_S10);
    msa_ldi_b(IMM_S10)
}

/// Immediate Load
///
/// The signed immediate imm_s10 is replicated in all vector
/// (eight signed 16-bit integer numbers) elements. For byte elements,
/// only the least significant 8 bits of imm_s10 will be used.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ldi.h, imm_s10 = 0b1111111111))]
#[rustc_legacy_const_generics(0)]
pub unsafe fn __msa_ldi_h<const IMM_S10: i32>() -> v8i16 {
    static_assert_imm_s10!(IMM_S10);
    msa_ldi_h(IMM_S10)
}

/// Immediate Load
///
/// The signed immediate imm_s10 is replicated in all vector
/// (four signed 32-bit integer numbers) elements. For byte elements,
/// only the least significant 8 bits of imm_s10 will be used.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ldi.w, imm_s10 = 0b1111111111))]
#[rustc_legacy_const_generics(0)]
pub unsafe fn __msa_ldi_w<const IMM_S10: i32>() -> v4i32 {
    static_assert_imm_s10!(IMM_S10);
    msa_ldi_w(IMM_S10)
}

/// Immediate Load
///
/// The signed immediate imm_s10 is replicated in all vector
/// (two signed 64-bit integer numbers) elements. For byte elements,
/// only the least significant 8 bits of imm_s10 will be used.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ldi.d, imm_s10 = 0b1111111111))]
#[rustc_legacy_const_generics(0)]
pub unsafe fn __msa_ldi_d<const IMM_S10: i32>() -> v2i64 {
    static_assert_imm_s10!(IMM_S10);
    msa_ldi_d(IMM_S10)
}

/// Vector Fixed-Point Multiply and Add
///
/// The products of fixed-point elements in `b` (eight signed 16-bit integer numbers)
/// by fixed-point elements in vector `c` (eight signed 16-bit integer numbers)
/// are added to the fixed-point elements in vector `a` (eight signed 16-bit integer numbers).
/// The multiplication result is not saturated, i.e. exact (-1) * (-1) = 1 is added to the destination.
/// The saturated fixed-point results are stored to vector `a`.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(madd_q.h))]
pub unsafe fn __msa_madd_q_h(a: v8i16, b: v8i16, c: v8i16) -> v8i16 {
    msa_madd_q_h(a, mem::transmute(b), c)
}

/// Vector Fixed-Point Multiply and Add
///
/// The products of fixed-point elements in `b` (four signed 32-bit integer numbers)
/// by fixed-point elements in vector `c` (four signed 32-bit integer numbers)
/// are added to the fixed-point elements in vector `a` (four signed 32-bit integer numbers).
/// The multiplication result is not saturated, i.e. exact (-1) * (-1) = 1 is added to the destination.
/// The saturated fixed-point results are stored to vector `a`.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(madd_q.w))]
pub unsafe fn __msa_madd_q_w(a: v4i32, b: v4i32, c: v4i32) -> v4i32 {
    msa_madd_q_w(a, mem::transmute(b), c)
}

/// Vector Fixed-Point Multiply and Add Rounded
///
/// The products of fixed-point elements in `b` (eight signed 16-bit integer numbers)
/// by fixed-point elements in vector `c` (eight signed 16-bit integer numbers)
/// are added to the fixed-point elements in vector `a` (eight signed 16-bit integer numbers).
/// The multiplication result is not saturated, i.e. exact (-1) * (-1) = 1 is added to the destination.
/// The rounded and saturated fixed-point results are stored to vector `a`.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(maddr_q.h))]
pub unsafe fn __msa_maddr_q_h(a: v8i16, b: v8i16, c: v8i16) -> v8i16 {
    msa_maddr_q_h(a, mem::transmute(b), c)
}

/// Vector Fixed-Point Multiply and Add Rounded
///
/// The products of fixed-point elements in `b` (four signed 32-bit integer numbers)
/// by fixed-point elements in vector `c` (four signed 32-bit integer numbers)
/// are added to the fixed-point elements in vector `a` (four signed 32-bit integer numbers).
/// The multiplication result is not saturated, i.e. exact (-1) * (-1) = 1 is added to the destination.
/// The rounded and saturated fixed-point results are stored to vector `a`.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(maddr_q.w))]
pub unsafe fn __msa_maddr_q_w(a: v4i32, b: v4i32, c: v4i32) -> v4i32 {
    msa_maddr_q_w(a, mem::transmute(b), c)
}

/// Vector Multiply and Add
///
/// The integer elements in vector `b` (sixteen signed 8-bit integer numbers)
/// are multiplied by integer elements in vector `c` (sixteen signed 8-bit integer numbers)
/// and added to the integer elements in vector `a` (sixteen signed 8-bit integer numbers).
/// The most significant half of the multiplication result is discarded.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(maddv.b))]
pub unsafe fn __msa_maddv_b(a: v16i8, b: v16i8, c: v16i8) -> v16i8 {
    msa_maddv_b(a, mem::transmute(b), c)
}

/// Vector Multiply and Add
///
/// The integer elements in vector `b` (eight signed 16-bit integer numbers)
/// are multiplied by integer elements in vector `c` (eight signed 16-bit integer numbers)
/// and added to the integer elements in vector `a` (eight signed 16-bit integer numbers).
/// The most significant half of the multiplication result is discarded.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(maddv.h))]
pub unsafe fn __msa_maddv_h(a: v8i16, b: v8i16, c: v8i16) -> v8i16 {
    msa_maddv_h(a, mem::transmute(b), c)
}

/// Vector Multiply and Add
///
/// The integer elements in vector `b` (four signed 32-bit integer numbers)
/// are multiplied by integer elements in vector `c` (four signed 32-bit integer numbers)
/// and added to the integer elements in vector `a` (four signed 32-bit integer numbers).
/// The most significant half of the multiplication result is discarded.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(maddv.w))]
pub unsafe fn __msa_maddv_w(a: v4i32, b: v4i32, c: v4i32) -> v4i32 {
    msa_maddv_w(a, mem::transmute(b), c)
}

/// Vector Multiply and Add
///
/// The integer elements in vector `b` (two signed 64-bit integer numbers)
/// are multiplied by integer elements in vector `c` (two signed 64-bit integer numbers)
/// and added to the integer elements in vector `a` (two signed 64-bit integer numbers).
/// The most significant half of the multiplication result is discarded.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(maddv.d))]
pub unsafe fn __msa_maddv_d(a: v2i64, b: v2i64, c: v2i64) -> v2i64 {
    msa_maddv_d(a, mem::transmute(b), c)
}

/// Vector Maximum Based on Absolute Values
///
/// The value with the largest magnitude, i.e. absolute value, between corresponding
/// signed elements in vector `a` (sixteen signed 8-bit integer numbers) and
/// `b` (sixteen signed 8-bit integer numbers) are written to vector
/// (sixteen signed 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(max_a.b))]
pub unsafe fn __msa_max_a_b(a: v16i8, b: v16i8) -> v16i8 {
    msa_max_a_b(a, mem::transmute(b))
}

/// Vector Maximum Based on Absolute Values
///
/// The value with the largest magnitude, i.e. absolute value, between corresponding
/// signed elements in vector `a` (eight signed 16-bit integer numbers) and
/// `b` (eight signed 16-bit integer numbers) are written to vector
/// (eight signed 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(max_a.h))]
pub unsafe fn __msa_max_a_h(a: v8i16, b: v8i16) -> v8i16 {
    msa_max_a_h(a, mem::transmute(b))
}

/// Vector Maximum Based on Absolute Values
///
/// The value with the largest magnitude, i.e. absolute value, between corresponding
/// signed elements in vector `a` (four signed 32-bit integer numbers) and
/// `b` (four signed 32-bit integer numbers) are written to vector
/// (four signed 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(max_a.w))]
pub unsafe fn __msa_max_a_w(a: v4i32, b: v4i32) -> v4i32 {
    msa_max_a_w(a, mem::transmute(b))
}

/// Vector Maximum Based on Absolute Values
///
/// The value with the largest magnitude, i.e. absolute value, between corresponding
/// signed elements in vector `a` (two signed 64-bit integer numbers) and
/// `b` (two signed 64-bit integer numbers) are written to vector
/// (two signed 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(max_a.d))]
pub unsafe fn __msa_max_a_d(a: v2i64, b: v2i64) -> v2i64 {
    msa_max_a_d(a, mem::transmute(b))
}

/// Vector Signed Maximum
///
/// Maximum values between signed elements in vector `a` (sixteen signed 8-bit integer numbers)
/// and signed elements in vector `b` (sixteen signed 8-bit integer numbers) are written to vector
/// (sixteen signed 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(max_s.b))]
pub unsafe fn __msa_max_s_b(a: v16i8, b: v16i8) -> v16i8 {
    msa_max_s_b(a, mem::transmute(b))
}

/// Vector Signed Maximum
///
/// Maximum values between signed elements in vector `a` (eight signed 16-bit integer numbers)
/// and signed elements in vector `b` (eight signed 16-bit integer numbers) are written to vector
/// (eight signed 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(max_s.h))]
pub unsafe fn __msa_max_s_h(a: v8i16, b: v8i16) -> v8i16 {
    msa_max_s_h(a, mem::transmute(b))
}

/// Vector Signed Maximum
///
/// Maximum values between signed elements in vector `a` (four signed 32-bit integer numbers)
/// and signed elements in vector `b` (four signed 32-bit integer numbers) are written to vector
/// (four signed 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(max_s.w))]
pub unsafe fn __msa_max_s_w(a: v4i32, b: v4i32) -> v4i32 {
    msa_max_s_w(a, mem::transmute(b))
}

/// Vector Signed Maximum
///
/// Maximum values between signed elements in vector `a` (two signed 64-bit integer numbers)
/// and signed elements in vector `b` (two signed 64-bit integer numbers) are written to vector
/// (two signed 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(max_s.d))]
pub unsafe fn __msa_max_s_d(a: v2i64, b: v2i64) -> v2i64 {
    msa_max_s_d(a, mem::transmute(b))
}

/// Vector Unsigned Maximum
///
/// Maximum values between unsigned elements in vector `a` (sixteen unsigned 8-bit integer numbers)
/// and unsigned elements in vector `b` (sixteen unsigned 8-bit integer numbers) are written to vector
/// (sixteen unsigned 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(max_u.b))]
pub unsafe fn __msa_max_u_b(a: v16u8, b: v16u8) -> v16u8 {
    msa_max_u_b(a, mem::transmute(b))
}

/// Vector Unsigned Maximum
///
/// Maximum values between unsigned elements in vector `a` (eight unsigned 16-bit integer numbers)
/// and unsigned elements in vector `b` (eight unsigned 16-bit integer numbers) are written to vector
/// (eight unsigned 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(max_u.h))]
pub unsafe fn __msa_max_u_h(a: v8u16, b: v8u16) -> v8u16 {
    msa_max_u_h(a, mem::transmute(b))
}

/// Vector Unsigned Maximum
///
/// Maximum values between unsigned elements in vector `a` (four unsigned 32-bit integer numbers)
/// and unsigned elements in vector `b` (four unsigned 32-bit integer numbers) are written to vector
/// (four unsigned 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(max_u.w))]
pub unsafe fn __msa_max_u_w(a: v4u32, b: v4u32) -> v4u32 {
    msa_max_u_w(a, mem::transmute(b))
}

/// Vector Unsigned Maximum
///
/// Maximum values between unsigned elements in vector `a` (two unsigned 64-bit integer numbers)
/// and unsigned elements in vector `b` (two unsigned 64-bit integer numbers) are written to vector
/// (two unsigned 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(max_u.d))]
pub unsafe fn __msa_max_u_d(a: v2u64, b: v2u64) -> v2u64 {
    msa_max_u_d(a, mem::transmute(b))
}

/// Immediate Signed Maximum
///
/// Maximum values between signed elements in vector `a` (sixteen signed 8-bit integer numbers)
/// and the 5-bit signed immediate imm_s5 are written to vector
/// (sixteen signed 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(maxi_s.b, imm5 = 0b11111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_maxi_s_b<const IMM_S5: i32>(a: v16i8) -> v16i8 {
    static_assert_imm_s5!(IMM_S5);
    msa_maxi_s_b(a, IMM_S5)
}

/// Immediate Signed Maximum
///
/// Maximum values between signed elements in vector `a` (eight signed 16-bit integer numbers)
/// and the 5-bit signed immediate imm_s5 are written to vector
/// (eight signed 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(maxi_s.h, imm_s5 = 0b11111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_maxi_s_h<const IMM_S5: i32>(a: v8i16) -> v8i16 {
    static_assert_imm_s5!(IMM_S5);
    msa_maxi_s_h(a, IMM_S5)
}

/// Immediate Signed Maximum
///
/// Maximum values between signed elements in vector `a` (four signed 32-bit integer numbers)
/// and the 5-bit signed immediate imm_s5 are written to vector
/// (four signed 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(maxi_s.w, imm_s5 = 0b11111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_maxi_s_w<const IMM_S5: i32>(a: v4i32) -> v4i32 {
    static_assert_imm_s5!(IMM_S5);
    msa_maxi_s_w(a, IMM_S5)
}

/// Immediate Signed Maximum
///
/// Maximum values between signed elements in vector `a` (two signed 64-bit integer numbers)
/// and the 5-bit signed immediate imm_s5 are written to vector
/// (two signed 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(maxi_s.d, imm_s5 = 0b11111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_maxi_s_d<const IMM_S5: i32>(a: v2i64) -> v2i64 {
    static_assert_imm_s5!(IMM_S5);
    msa_maxi_s_d(a, IMM_S5)
}

/// Immediate Unsigned Maximum
///
/// Maximum values between unsigned elements in vector `a` (sixteen unsigned 8-bit integer numbers)
/// and the 5-bit unsigned immediate `imm5` are written to vector
/// (sixteen unsigned 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(maxi_u.b, imm5 = 0b11111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_maxi_u_b<const IMM5: i32>(a: v16u8) -> v16u8 {
    static_assert_imm5!(IMM5);
    msa_maxi_u_b(a, IMM5)
}

/// Immediate Unsigned Maximum
///
/// Maximum values between unsigned elements in vector `a` (eight unsigned 16-bit integer numbers)
/// and the 5-bit unsigned immediate `imm5` are written to vector
/// (eight unsigned 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(maxi_u.h, imm5 = 0b11111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_maxi_u_h<const IMM5: i32>(a: v8u16) -> v8u16 {
    static_assert_imm5!(IMM5);
    msa_maxi_u_h(a, IMM5)
}

/// Immediate Unsigned Maximum
///
/// Maximum values between unsigned elements in vector `a` (four unsigned 32-bit integer numbers)
/// and the 5-bit unsigned immediate `imm5` are written to vector
/// (four unsigned 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(maxi_u.w, imm5 = 0b11111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_maxi_u_w<const IMM5: i32>(a: v4u32) -> v4u32 {
    static_assert_imm5!(IMM5);
    msa_maxi_u_w(a, IMM5)
}

/// Immediate Unsigned Maximum
///
/// Maximum values between unsigned elements in vector `a` (two unsigned 64-bit integer numbers)
/// and the 5-bit unsigned immediate `imm5` are written to vector
/// (two unsigned 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(maxi_u.d, imm5 = 0b11111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_maxi_u_d<const IMM5: i32>(a: v2u64) -> v2u64 {
    static_assert_imm5!(IMM5);
    msa_maxi_u_d(a, IMM5)
}

/// Vector Minimum Based on Absolute Value
///
/// The value with the smallest magnitude, i.e. absolute value, between corresponding
/// signed elements in vector `a` (sixteen signed 8-bit integer numbers) and
/// `b` (sixteen signed 8-bit integer numbers) are written to vector
/// (sixteen signed 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(min_a.b))]
pub unsafe fn __msa_min_a_b(a: v16i8, b: v16i8) -> v16i8 {
    msa_min_a_b(a, mem::transmute(b))
}

/// Vector Minimum Based on Absolute Value
///
/// The value with the smallest magnitude, i.e. absolute value, between corresponding
/// signed elements in vector `a` (eight signed 16-bit integer numbers) and
/// `b` (eight signed 16-bit integer numbers) are written to vector
/// (eight signed 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(min_a.h))]
pub unsafe fn __msa_min_a_h(a: v8i16, b: v8i16) -> v8i16 {
    msa_min_a_h(a, mem::transmute(b))
}

/// Vector Minimum Based on Absolute Value
///
/// The value with the smallest magnitude, i.e. absolute value, between corresponding
/// signed elements in vector `a` (four signed 32-bit integer numbers) and
/// `b` (four signed 32-bit integer numbers) are written to vector
/// (four signed 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(min_a.w))]
pub unsafe fn __msa_min_a_w(a: v4i32, b: v4i32) -> v4i32 {
    msa_min_a_w(a, mem::transmute(b))
}

/// Vector Minimum Based on Absolute Value
///
/// The value with the smallest magnitude, i.e. absolute value, between corresponding
/// signed elements in vector `a` (two signed 64-bit integer numbers) and
/// `b` (two signed 64-bit integer numbers) are written to vector
/// (two signed 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(min_a.d))]
pub unsafe fn __msa_min_a_d(a: v2i64, b: v2i64) -> v2i64 {
    msa_min_a_d(a, mem::transmute(b))
}

/// Vector Signed Minimum
///
/// Minimum values between signed elements in vector `a` (sixteen signed 8-bit integer numbers)
/// and signed elements in vector `b` (sixteen signed 8-bit integer numbers) are written to vector
/// (sixteen signed 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(min_s.b))]
pub unsafe fn __msa_min_s_b(a: v16i8, b: v16i8) -> v16i8 {
    msa_min_s_b(a, mem::transmute(b))
}

/// Vector Signed Minimum
///
/// Minimum values between signed elements in vector `a` (eight signed 16-bit integer numbers)
/// and signed elements in vector `b` (eight signed 16-bit integer numbers) are written to vector
/// (eight signed 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(min_s.h))]
pub unsafe fn __msa_min_s_h(a: v8i16, b: v8i16) -> v8i16 {
    msa_min_s_h(a, mem::transmute(b))
}

/// Vector Signed Minimum
///
/// Minimum values between signed elements in vector `a` (four signed 32-bit integer numbers)
/// and signed elements in vector `b` (four signed 32-bit integer numbers) are written to vector
/// (four signed 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(min_s.w))]
pub unsafe fn __msa_min_s_w(a: v4i32, b: v4i32) -> v4i32 {
    msa_min_s_w(a, mem::transmute(b))
}

/// Vector Signed Minimum
///
/// Minimum values between signed elements in vector `a` (two signed 64-bit integer numbers)
/// and signed elements in vector `b` (two signed 64-bit integer numbers) are written to vector
/// (two signed 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(min_s.d))]
pub unsafe fn __msa_min_s_d(a: v2i64, b: v2i64) -> v2i64 {
    msa_min_s_d(a, mem::transmute(b))
}

/// Immediate Signed Minimum
///
/// Minimum values between signed elements in vector `a` (sixteen signed 8-bit integer numbers)
/// and the 5-bit signed immediate imm_s5 are written to vector
/// (sixteen signed 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(mini_s.b, imm_s5 = 0b11111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_mini_s_b<const IMM_S5: i32>(a: v16i8) -> v16i8 {
    static_assert_imm_s5!(IMM_S5);
    msa_mini_s_b(a, IMM_S5)
}

/// Immediate Signed Minimum
///
/// Minimum values between signed elements in vector `a` (eight signed 16-bit integer numbers)
/// and the 5-bit signed immediate imm_s5 are written to vector
/// (eight signed 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(mini_s.h, imm_s5 = 0b11111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_mini_s_h<const IMM_S5: i32>(a: v8i16) -> v8i16 {
    static_assert_imm_s5!(IMM_S5);
    msa_mini_s_h(a, IMM_S5)
}

/// Immediate Signed Minimum
///
/// Minimum values between signed elements in vector `a` (four signed 32-bit integer numbers)
/// and the 5-bit signed immediate imm_s5 are written to vector
/// (four signed 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(mini_s.w, imm_s5 = 0b11111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_mini_s_w<const IMM_S5: i32>(a: v4i32) -> v4i32 {
    static_assert_imm_s5!(IMM_S5);
    msa_mini_s_w(a, IMM_S5)
}

/// Immediate Signed Minimum
///
/// Minimum values between signed elements in vector `a` (two signed 64-bit integer numbers)
/// and the 5-bit signed immediate imm_s5 are written to vector
/// (two signed 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(mini_s.d, imm_s5 = 0b11111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_mini_s_d<const IMM_S5: i32>(a: v2i64) -> v2i64 {
    static_assert_imm_s5!(IMM_S5);
    msa_mini_s_d(a, IMM_S5)
}

/// Vector Unsigned Minimum
///
/// Minimum values between unsigned elements in vector `a` (sixteen unsigned 8-bit integer numbers)
/// and unsigned elements in vector `b` (sixteen unsigned 8-bit integer numbers) are written to vector
/// (sixteen unsigned 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(min_u.b))]
pub unsafe fn __msa_min_u_b(a: v16u8, b: v16u8) -> v16u8 {
    msa_min_u_b(a, mem::transmute(b))
}

/// Vector Unsigned Minimum
///
/// Minimum values between unsigned elements in vector `a` (eight unsigned 16-bit integer numbers)
/// and unsigned elements in vector `b` (eight unsigned 16-bit integer numbers) are written to vector
/// (eight unsigned 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(min_u.h))]
pub unsafe fn __msa_min_u_h(a: v8u16, b: v8u16) -> v8u16 {
    msa_min_u_h(a, mem::transmute(b))
}

/// Vector Unsigned Minimum
///
/// Minimum values between unsigned elements in vector `a` (four unsigned 32-bit integer numbers)
/// and unsigned elements in vector `b` (four unsigned 32-bit integer numbers) are written to vector
/// (four unsigned 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(min_u.w))]
pub unsafe fn __msa_min_u_w(a: v4u32, b: v4u32) -> v4u32 {
    msa_min_u_w(a, mem::transmute(b))
}

/// Vector Unsigned Minimum
///
/// Minimum values between unsigned elements in vector `a` (two unsigned 64-bit integer numbers)
/// and unsigned elements in vector `b` (two unsigned 64-bit integer numbers) are written to vector
/// (two unsigned 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(min_u.d))]
pub unsafe fn __msa_min_u_d(a: v2u64, b: v2u64) -> v2u64 {
    msa_min_u_d(a, mem::transmute(b))
}

/// Immediate Unsigned Minimum
///
/// Minimum values between unsigned elements in vector `a` (sixteen unsigned 8-bit integer numbers)
/// and the 5-bit unsigned immediate `imm5` are written to vector
/// (sixteen unsigned 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(mini_u.b, imm5 = 0b11111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_mini_u_b<const IMM5: i32>(a: v16u8) -> v16u8 {
    static_assert_imm5!(IMM5);
    msa_mini_u_b(a, IMM5)
}

/// Immediate Unsigned Minimum
///
/// Minimum values between unsigned elements in vector `a` (eight unsigned 16-bit integer numbers)
/// and the 5-bit unsigned immediate `imm5` are written to vector
/// (eight unsigned 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(mini_u.h, imm5 = 0b11111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_mini_u_h<const IMM5: i32>(a: v8u16) -> v8u16 {
    static_assert_imm5!(IMM5);
    msa_mini_u_h(a, IMM5)
}

/// Immediate Unsigned Minimum
///
/// Minimum values between unsigned elements in vector `a` (four unsigned 32-bit integer numbers)
/// and the 5-bit unsigned immediate `imm5` are written to vector
/// (four unsigned 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(mini_u.w, imm5 = 0b11111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_mini_u_w<const IMM5: i32>(a: v4u32) -> v4u32 {
    static_assert_imm5!(IMM5);
    msa_mini_u_w(a, IMM5)
}

/// Immediate Unsigned Minimum
///
/// Minimum values between unsigned elements in vector `a` (two unsigned 64-bit integer numbers)
/// and the 5-bit unsigned immediate `imm5` are written to vector
/// (two unsigned 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(mini_u.d, imm5 = 0b11111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_mini_u_d<const IMM5: i32>(a: v2u64) -> v2u64 {
    static_assert_imm5!(IMM5);
    msa_mini_u_d(a, IMM5)
}

/// Vector Signed Modulo
///
/// The signed integer elements in vector `a` (sixteen signed 8-bit integer numbers)
/// are divided by signed integer elements in vector `b` (sixteen signed 8-bit integer numbers).
/// The remainder of the same sign as the dividend is written to vector
/// (sixteen signed 8-bit integer numbers). If a divisor element vector `b` is zero,
/// the result value is UNPREDICTABLE.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(mod_s.b))]
pub unsafe fn __msa_mod_s_b(a: v16i8, b: v16i8) -> v16i8 {
    msa_mod_s_b(a, mem::transmute(b))
}

/// Vector Signed Modulo
///
/// The signed integer elements in vector `a` (eight signed 16-bit integer numbers)
/// are divided by signed integer elements in vector `b` (eight signed 16-bit integer numbers).
/// The remainder of the same sign as the dividend is written to vector
/// (eight signed 16-bit integer numbers). If a divisor element vector `b` is zero,
/// the result value is UNPREDICTABLE.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(mod_s.h))]
pub unsafe fn __msa_mod_s_h(a: v8i16, b: v8i16) -> v8i16 {
    msa_mod_s_h(a, mem::transmute(b))
}

/// Vector Signed Modulo
///
/// The signed integer elements in vector `a` (four signed 32-bit integer numbers)
/// are divided by signed integer elements in vector `b` (four signed 32-bit integer numbers).
/// The remainder of the same sign as the dividend is written to vector
/// (four signed 32-bit integer numbers). If a divisor element vector `b` is zero,
/// the result value is UNPREDICTABLE.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(mod_s.w))]
pub unsafe fn __msa_mod_s_w(a: v4i32, b: v4i32) -> v4i32 {
    msa_mod_s_w(a, mem::transmute(b))
}

/// Vector Signed Modulo
///
/// The signed integer elements in vector `a` (two signed 64-bit integer numbers)
/// are divided by signed integer elements in vector `b` (two signed 64-bit integer numbers).
/// The remainder of the same sign as the dividend is written to vector
/// (two signed 64-bit integer numbers). If a divisor element vector `b` is zero,
/// the result value is UNPREDICTABLE.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(mod_s.d))]
pub unsafe fn __msa_mod_s_d(a: v2i64, b: v2i64) -> v2i64 {
    msa_mod_s_d(a, mem::transmute(b))
}

/// Vector Unsigned Modulo
///
/// The unsigned integer elements in vector `a` (sixteen unsigned 8-bit integer numbers)
/// are divided by unsigned integer elements in vector `b` (sixteen unsigned 8-bit integer numbers).
/// The remainder of the same sign as the dividend is written to vector
/// (sixteen unsigned 8-bit integer numbers). If a divisor element vector `b` is zero,
/// the result value is UNPREDICTABLE.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(mod_u.b))]
pub unsafe fn __msa_mod_u_b(a: v16u8, b: v16u8) -> v16u8 {
    msa_mod_u_b(a, mem::transmute(b))
}

/// Vector Unsigned Modulo
///
/// The unsigned integer elements in vector `a` (eight unsigned 16-bit integer numbers)
/// are divided by unsigned integer elements in vector `b` (eight unsigned 16-bit integer numbers).
/// The remainder of the same sign as the dividend is written to vector
/// (eight unsigned 16-bit integer numbers). If a divisor element vector `b` is zero,
/// the result value is UNPREDICTABLE.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(mod_u.h))]
pub unsafe fn __msa_mod_u_h(a: v8u16, b: v8u16) -> v8u16 {
    msa_mod_u_h(a, mem::transmute(b))
}

/// Vector Unsigned Modulo
///
/// The unsigned integer elements in vector `a` (four unsigned 32-bit integer numbers)
/// are divided by unsigned integer elements in vector `b` (four unsigned 32-bit integer numbers).
/// The remainder of the same sign as the dividend is written to vector
/// (four unsigned 32-bit integer numbers). If a divisor element vector `b` is zero,
/// the result value is UNPREDICTABLE.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(mod_u.w))]
pub unsafe fn __msa_mod_u_w(a: v4u32, b: v4u32) -> v4u32 {
    msa_mod_u_w(a, mem::transmute(b))
}

/// Vector Unsigned Modulo
///
/// The unsigned integer elements in vector `a` (two unsigned 64-bit integer numbers)
/// are divided by unsigned integer elements in vector `b` (two unsigned 64-bit integer numbers).
/// The remainder of the same sign as the dividend is written to vector
/// (two unsigned 64-bit integer numbers). If a divisor element vector `b` is zero,
/// the result value is UNPREDICTABLE.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(mod_u.d))]
pub unsafe fn __msa_mod_u_d(a: v2u64, b: v2u64) -> v2u64 {
    msa_mod_u_d(a, mem::transmute(b))
}

/// Vector Move
///
/// Copy all WRLEN bits in vector `a` (eight signed 16-bit integer numbers)
/// to vector (eight signed 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(move.v))]
pub unsafe fn __msa_move_v(a: v16i8) -> v16i8 {
    msa_move_v(a)
}

/// Vector Fixed-Point Multiply and Subtract
///
/// The product of fixed-point elements in vector `c` (eight signed 16-bit integer numbers)
/// by fixed-point elements in vector `b` (eight signed 16-bit integer numbers)
/// are subtracted from the fixed-point elements in vector `a`
/// (eight signed 16-bit integer numbers). The multiplication result is not saturated,
/// i.e. exact (-1) * (-1) = 1 is subtracted from the destination.
/// The saturated fixed-point results are stored back to vector `a`.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(msub_q.h))]
pub unsafe fn __msa_msub_q_h(a: v8i16, b: v8i16, c: v8i16) -> v8i16 {
    msa_msub_q_h(a, mem::transmute(b), c)
}

/// Vector Fixed-Point Multiply and Subtract
///
/// The product of fixed-point elements in vector `c` (four signed 32-bit integer numbers)
/// by fixed-point elements in vector `b` (four signed 32-bit integer numbers)
/// are subtracted from the fixed-point elements in vector `a`
/// (four signed 32-bit integer numbers). The multiplication result is not saturated,
/// i.e. exact (-1) * (-1) = 1 is subtracted from the destination.
/// The saturated fixed-point results are stored back to vector `a`.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(msub_q.w))]
pub unsafe fn __msa_msub_q_w(a: v4i32, b: v4i32, c: v4i32) -> v4i32 {
    msa_msub_q_w(a, mem::transmute(b), c)
}

/// Vector Fixed-Point Multiply and Subtract Rounded
///
/// The product of fixed-point elements in vector `c` (eight signed 16-bit integer numbers)
/// by fixed-point elements in vector `b` (eight signed 16-bit integer numbers)
/// are subtracted from the fixed-point elements in vector `a`
/// (eight signed 16-bit integer numbers). The multiplication result is not saturated,
/// i.e. exact (-1) * (-1) = 1 is subtracted from the destination.
/// The rounded and saturated fixed-point results are stored back to vector `a`.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(msubr_q.h))]
pub unsafe fn __msa_msubr_q_h(a: v8i16, b: v8i16, c: v8i16) -> v8i16 {
    msa_msubr_q_h(a, mem::transmute(b), c)
}

/// Vector Fixed-Point Multiply and Subtract Rounded
///
/// The product of fixed-point elements in vector `c` (four signed 32-bit integer numbers)
/// by fixed-point elements in vector `b` (four signed 32-bit integer numbers)
/// are subtracted from the fixed-point elements in vector `a`
/// (four signed 32-bit integer numbers). The multiplication result is not saturated,
/// i.e. exact (-1) * (-1) = 1 is subtracted from the destination.
/// The rounded and saturated fixed-point results are stored back to vector `a`.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(msubr_q.w))]
pub unsafe fn __msa_msubr_q_w(a: v4i32, b: v4i32, c: v4i32) -> v4i32 {
    msa_msubr_q_w(a, mem::transmute(b), c)
}

/// Vector Multiply and Subtract
///
/// The integer elements in vector `c` (sixteen signed 8-bit integer numbers)
/// are multiplied by integer elements in vector `b` (sixteen signed 8-bit integer numbers)
/// and subtracted from the integer elements in vector `a` (sixteen signed 8-bit integer numbers).
/// The most significant half of the multiplication result is discarded.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(msubv.b))]
pub unsafe fn __msa_msubv_b(a: v16i8, b: v16i8, c: v16i8) -> v16i8 {
    msa_msubv_b(a, mem::transmute(b), c)
}

/// Vector Multiply and Subtract
///
/// The integer elements in vector `c` (eight signed 16-bit integer numbers)
/// are multiplied by integer elements in vector `b` (eight signed 16-bit integer numbers)
/// and subtracted from the integer elements in vector `a` (eight signed 16-bit integer numbers).
/// The most significant half of the multiplication result is discarded.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(msubv.h))]
pub unsafe fn __msa_msubv_h(a: v8i16, b: v8i16, c: v8i16) -> v8i16 {
    msa_msubv_h(a, mem::transmute(b), c)
}

/// Vector Multiply and Subtract
///
/// The integer elements in vector `c` (four signed 32-bit integer numbers)
/// are multiplied by integer elements in vector `b` (four signed 32-bit integer numbers)
/// and subtracted from the integer elements in vector `a` (four signed 32-bit integer numbers).
/// The most significant half of the multiplication result is discarded.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(msubv.w))]
pub unsafe fn __msa_msubv_w(a: v4i32, b: v4i32, c: v4i32) -> v4i32 {
    msa_msubv_w(a, mem::transmute(b), c)
}

/// Vector Multiply and Subtract
///
/// The integer elements in vector `c` (two signed 64-bit integer numbers)
/// are multiplied by integer elements in vector `b` (two signed 64-bit integer numbers)
/// and subtracted from the integer elements in vector `a` (two signed 64-bit integer numbers).
/// The most significant half of the multiplication result is discarded.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(msubv.d))]
pub unsafe fn __msa_msubv_d(a: v2i64, b: v2i64, c: v2i64) -> v2i64 {
    msa_msubv_d(a, mem::transmute(b), c)
}

/// Vector Fixed-Point Multiply
///
/// The fixed-point elements in vector `a` (eight signed 16-bit integer numbers)
/// multiplied by fixed-point elements in vector `b` (eight signed 16-bit integer numbers).
/// The result is written to vector (eight signed 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(mul_q.h))]
pub unsafe fn __msa_mul_q_h(a: v8i16, b: v8i16) -> v8i16 {
    msa_mul_q_h(a, mem::transmute(b))
}

/// Vector Fixed-Point Multiply
///
/// The fixed-point elements in vector `a` (four signed 32-bit integer numbers)
/// multiplied by fixed-point elements in vector `b` (four signed 32-bit integer numbers).
/// The result is written to vector (four signed 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(mul_q.w))]
pub unsafe fn __msa_mul_q_w(a: v4i32, b: v4i32) -> v4i32 {
    msa_mul_q_w(a, mem::transmute(b))
}

/// Vector Fixed-Point Multiply Rounded
///
/// The fixed-point elements in vector `a` (eight signed 16-bit integer numbers)
/// multiplied by fixed-point elements in vector `b` (eight signed 16-bit integer numbers).
/// The rounded result is written to vector (eight signed 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(mulr_q.h))]
pub unsafe fn __msa_mulr_q_h(a: v8i16, b: v8i16) -> v8i16 {
    msa_mulr_q_h(a, mem::transmute(b))
}

/// Vector Fixed-Point Multiply Rounded
///
/// The fixed-point elements in vector `a` (four signed 32-bit integer numbers)
/// multiplied by fixed-point elements in vector `b` (four signed 32-bit integer numbers).
/// The rounded result is written to vector (four signed 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(mulr_q.w))]
pub unsafe fn __msa_mulr_q_w(a: v4i32, b: v4i32) -> v4i32 {
    msa_mulr_q_w(a, mem::transmute(b))
}

/// Vector Multiply
///
/// The integer elements in vector `a` (sixteen signed 8-bit integer numbers)
/// are multiplied by integer elements in vector `b` (sixteen signed 8-bit integer numbers).
/// The result is written to vector (sixteen signed 8-bit integer numbers).
/// The most significant half of the multiplication result is discarded.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(mulv.b))]
pub unsafe fn __msa_mulv_b(a: v16i8, b: v16i8) -> v16i8 {
    msa_mulv_b(a, mem::transmute(b))
}

/// Vector Multiply
///
/// The integer elements in vector `a` (eight signed 16-bit integer numbers)
/// are multiplied by integer elements in vector `b` (eight signed 16-bit integer numbers).
/// The result is written to vector (eight signed 16-bit integer numbers).
/// The most significant half of the multiplication result is discarded.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(mulv.h))]
pub unsafe fn __msa_mulv_h(a: v8i16, b: v8i16) -> v8i16 {
    msa_mulv_h(a, mem::transmute(b))
}

/// Vector Multiply
///
/// The integer elements in vector `a` (four signed 32-bit integer numbers)
/// are multiplied by integer elements in vector `b` (four signed 32-bit integer numbers).
/// The result is written to vector (four signed 32-bit integer numbers).
/// The most significant half of the multiplication result is discarded.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(mulv.w))]
pub unsafe fn __msa_mulv_w(a: v4i32, b: v4i32) -> v4i32 {
    msa_mulv_w(a, mem::transmute(b))
}

/// Vector Multiply
///
/// The integer elements in vector `a` (two signed 64-bit integer numbers)
/// are multiplied by integer elements in vector `b` (two signed 64-bit integer numbers).
/// The result is written to vector (two signed 64-bit integer numbers).
/// The most significant half of the multiplication result is discarded.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(mulv.d))]
pub unsafe fn __msa_mulv_d(a: v2i64, b: v2i64) -> v2i64 {
    msa_mulv_d(a, mem::transmute(b))
}

/// Vector Leading Ones Count
///
/// The number of leading ones for elements in vector `a` (sixteen signed 8-bit integer numbers)
/// is stored to the elements in vector (sixteen signed 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(nloc.b))]
pub unsafe fn __msa_nloc_b(a: v16i8) -> v16i8 {
    msa_nloc_b(a)
}

/// Vector Leading Ones Count
///
/// The number of leading ones for elements in vector `a` (eight signed 16-bit integer numbers)
/// is stored to the elements in vector (eight signed 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(nloc.h))]
pub unsafe fn __msa_nloc_h(a: v8i16) -> v8i16 {
    msa_nloc_h(a)
}

/// Vector Leading Ones Count
///
/// The number of leading ones for elements in vector `a` (four signed 32-bit integer numbers)
/// is stored to the elements in vector (four signed 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(nloc.w))]
pub unsafe fn __msa_nloc_w(a: v4i32) -> v4i32 {
    msa_nloc_w(a)
}

/// Vector Leading Ones Count
///
/// The number of leading ones for elements in vector `a` (two signed 64-bit integer numbers)
/// is stored to the elements in vector (two signed 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(nloc.d))]
pub unsafe fn __msa_nloc_d(a: v2i64) -> v2i64 {
    msa_nloc_d(a)
}

/// Vector Leading Zeros Count
///
/// The number of leading zeros for elements in vector `a` (sixteen signed 8-bit integer numbers)
/// is stored to the elements in vector (sixteen signed 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(nlzc.b))]
pub unsafe fn __msa_nlzc_b(a: v16i8) -> v16i8 {
    msa_nlzc_b(a)
}

/// Vector Leading Zeros Count
///
/// The number of leading zeros for elements in vector `a` (eight signed 16-bit integer numbers)
/// is stored to the elements in vector (eight signed 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(nlzc.h))]
pub unsafe fn __msa_nlzc_h(a: v8i16) -> v8i16 {
    msa_nlzc_h(a)
}

/// Vector Leading Zeros Count
///
/// The number of leading zeros for elements in vector `a` (four signed 32-bit integer numbers)
/// is stored to the elements in vector (four signed 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(nlzc.w))]
pub unsafe fn __msa_nlzc_w(a: v4i32) -> v4i32 {
    msa_nlzc_w(a)
}

/// Vector Leading Zeros Count
///
/// The number of leading zeros for elements in vector `a` (two signed 64-bit integer numbers)
/// is stored to the elements in vector (two signed 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(nlzc.d))]
pub unsafe fn __msa_nlzc_d(a: v2i64) -> v2i64 {
    msa_nlzc_d(a)
}

/// Vector Logical Negated Or
///
/// Each bit of vector `a` (sixteen unsigned 8-bit integer numbers)
/// is combined with the corresponding bit of vector `b` (sixteen unsigned 8-bit integer numbers)
/// in a bitwise logical NOR operation. The result is written to vector
/// (sixteen unsigned 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(nor.v))]
pub unsafe fn __msa_nor_v(a: v16u8, b: v16u8) -> v16u8 {
    msa_nor_v(a, mem::transmute(b))
}

/// Immediate Logical Negated Or
///
/// Each bit of vector `a` (sixteen unsigned 8-bit integer numbers)
/// is combined with the 8-bit immediate `imm8`
/// in a bitwise logical NOR operation. The result is written to vector
/// (sixteen unsigned 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(nori.b, imm8 = 0b11111111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_nori_b<const IMM8: i32>(a: v16u8) -> v16u8 {
    static_assert_imm8!(IMM8);
    msa_nori_b(a, IMM8)
}

/// Vector Logical Or
///
/// Each bit of vector `a` (sixteen unsigned 8-bit integer numbers)
/// is combined with the corresponding bit of vector `b` (sixteen unsigned 8-bit integer numbers)
/// in a bitwise logical OR operation. The result is written to vector
/// (sixteen unsigned 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(or.v))]
pub unsafe fn __msa_or_v(a: v16u8, b: v16u8) -> v16u8 {
    msa_or_v(a, mem::transmute(b))
}

/// Immediate Logical Or
///
/// Each bit of vector `a` (sixteen unsigned 8-bit integer numbers)
/// is combined with the 8-bit immediate `imm8`
/// in a bitwise logical OR operation. The result is written to vector
/// (sixteen unsigned 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(ori.b, imm8 = 0b11111111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_ori_b<const IMM8: i32>(a: v16u8) -> v16u8 {
    static_assert_imm8!(IMM8);
    msa_ori_b(a, IMM8)
}

/// Vector Pack Even
///
/// Even elements in vectors `a` (sixteen signed 8-bit integer numbers)
/// are copied to the left half of the result vector and even elements in vector `b`
/// (sixteen signed 8-bit integer numbers) are copied to the right half of the result vector.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(pckev.b))]
pub unsafe fn __msa_pckev_b(a: v16i8, b: v16i8) -> v16i8 {
    msa_pckev_b(a, mem::transmute(b))
}

/// Vector Pack Even
///
/// Even elements in vectors `a` (eight signed 16-bit integer numbers)
/// are copied to the left half of the result vector and even elements in vector `b`
/// (eight signed 16-bit integer numbers) are copied to the right half of the result vector.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(pckev.h))]
pub unsafe fn __msa_pckev_h(a: v8i16, b: v8i16) -> v8i16 {
    msa_pckev_h(a, mem::transmute(b))
}

/// Vector Pack Even
///
/// Even elements in vectors `a` (four signed 32-bit integer numbers)
/// are copied to the left half of the result vector and even elements in vector `b`
/// (four signed 32-bit integer numbers) are copied to the right half of the result vector.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(pckev.w))]
pub unsafe fn __msa_pckev_w(a: v4i32, b: v4i32) -> v4i32 {
    msa_pckev_w(a, mem::transmute(b))
}

/// Vector Pack Even
///
/// Even elements in vectors `a` (two signed 64-bit integer numbers)
/// are copied to the left half of the result vector and even elements in vector `b`
/// (two signed 64-bit integer numbers) are copied to the right half of the result vector.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(pckev.d))]
pub unsafe fn __msa_pckev_d(a: v2i64, b: v2i64) -> v2i64 {
    msa_pckev_d(a, mem::transmute(b))
}

/// Vector Pack Odd
///
/// Odd elements in vectors `a` (sixteen signed 8-bit integer numbers)
/// are copied to the left half of the result vector and odd elements in vector `b`
/// (sixteen signed 8-bit integer numbers) are copied to the right half of the result vector.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(pckod.b))]
pub unsafe fn __msa_pckod_b(a: v16i8, b: v16i8) -> v16i8 {
    msa_pckod_b(a, mem::transmute(b))
}

/// Vector Pack Odd
///
/// Odd elements in vectors `a` (eight signed 16-bit integer numbers)
/// are copied to the left half of the result vector and odd elements in vector `b`
/// (eight signed 16-bit integer numbers) are copied to the right half of the result vector.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(pckod.h))]
pub unsafe fn __msa_pckod_h(a: v8i16, b: v8i16) -> v8i16 {
    msa_pckod_h(a, mem::transmute(b))
}

/// Vector Pack Odd
///
/// Odd elements in vectors `a` (four signed 32-bit integer numbers)
/// are copied to the left half of the result vector and odd elements in vector `b`
/// (four signed 32-bit integer numbers) are copied to the right half of the result vector.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(pckod.w))]
pub unsafe fn __msa_pckod_w(a: v4i32, b: v4i32) -> v4i32 {
    msa_pckod_w(a, mem::transmute(b))
}

/// Vector Pack Odd
///
/// Odd elements in vectors `a` (two signed 64-bit integer numbers)
/// are copied to the left half of the result vector and odd elements in vector `b`
/// (two signed 64-bit integer numbers) are copied to the right half of the result vector.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(pckod.d))]
pub unsafe fn __msa_pckod_d(a: v2i64, b: v2i64) -> v2i64 {
    msa_pckod_d(a, mem::transmute(b))
}

/// Vector Population Count
///
/// The number of bits set to 1 for elements in vector `a` (sixteen signed 8-bit integer numbers)
/// is stored to the elements in the result vector (sixteen signed 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(pcnt.b))]
pub unsafe fn __msa_pcnt_b(a: v16i8) -> v16i8 {
    msa_pcnt_b(a)
}

/// Vector Population Count
///
/// The number of bits set to 1 for elements in vector `a` (eight signed 16-bit integer numbers)
/// is stored to the elements in the result vector (eight signed 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(pcnt.h))]
pub unsafe fn __msa_pcnt_h(a: v8i16) -> v8i16 {
    msa_pcnt_h(a)
}

/// Vector Population Count
///
/// The number of bits set to 1 for elements in vector `a` (four signed 32-bit integer numbers)
/// is stored to the elements in the result vector (four signed 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(pcnt.w))]
pub unsafe fn __msa_pcnt_w(a: v4i32) -> v4i32 {
    msa_pcnt_w(a)
}

/// Vector Population Count
///
/// The number of bits set to 1 for elements in vector `a` (two signed 64-bit integer numbers)
/// is stored to the elements in the result vector (two signed 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(pcnt.d))]
pub unsafe fn __msa_pcnt_d(a: v2i64) -> v2i64 {
    msa_pcnt_d(a)
}

/// Immediate Signed Saturate
///
/// Signed elements in vector `a` (sixteen signed 8-bit integer numbers)
/// are saturated to signed values of `imm3+1` bits without changing the data width.
/// The result is stored in the vector (sixteen signed 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(sat_s.b, imm4 = 0b111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_sat_s_b<const IMM3: i32>(a: v16i8) -> v16i8 {
    static_assert_imm3!(IMM3);
    msa_sat_s_b(a, IMM3)
}

/// Immediate Signed Saturate
///
/// Signed elements in vector `a` (eight signed 16-bit integer numbers)
/// are saturated to signed values of `imm4+1` bits without changing the data width.
/// The result is stored in the vector (eight signed 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(sat_s.h, imm3 = 0b1111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_sat_s_h<const IMM4: i32>(a: v8i16) -> v8i16 {
    static_assert_imm4!(IMM4);
    msa_sat_s_h(a, IMM4)
}

/// Immediate Signed Saturate
///
/// Signed elements in vector `a` (four signed 32-bit integer numbers)
/// are saturated to signed values of `imm5+1` bits without changing the data width.
/// The result is stored in the vector (four signed 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(sat_s.w, imm2 = 0b11111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_sat_s_w<const IMM5: i32>(a: v4i32) -> v4i32 {
    static_assert_imm5!(IMM5);
    msa_sat_s_w(a, IMM5)
}

/// Immediate Signed Saturate
///
/// Signed elements in vector `a` (two signed 64-bit integer numbers)
/// are saturated to signed values of `imm6+1` bits without changing the data width.
/// The result is stored in the vector (two signed 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(sat_s.d, imm1 = 0b111111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_sat_s_d<const IMM6: i32>(a: v2i64) -> v2i64 {
    static_assert_imm6!(IMM6);
    msa_sat_s_d(a, IMM6)
}

/// Immediate Unsigned Saturate
///
/// Unsigned elements in vector `a` (sixteen unsigned 8-bit integer numbers)
/// are saturated to unsigned values of `imm3+1` bits without changing the data width.
/// The result is stored in the vector (sixteen unsigned 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(sat_u.b, imm4 = 0b111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_sat_u_b<const IMM3: i32>(a: v16u8) -> v16u8 {
    static_assert_imm3!(IMM3);
    msa_sat_u_b(a, IMM3)
}

/// Immediate Unsigned Saturate
///
/// Unsigned elements in vector `a` (eight unsigned 16-bit integer numbers)
/// are saturated to unsigned values of `imm4+1` bits without changing the data width.
/// The result is stored in the vector (eight unsigned 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(sat_u.h, imm3 = 0b1111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_sat_u_h<const IMM4: i32>(a: v8u16) -> v8u16 {
    static_assert_imm4!(IMM4);
    msa_sat_u_h(a, IMM4)
}

/// Immediate Unsigned Saturate
///
/// Unsigned elements in vector `a` (four unsigned 32-bit integer numbers)
/// are saturated to unsigned values of `imm5+1` bits without changing the data width.
/// The result is stored in the vector (four unsigned 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(sat_u.w, imm2 = 0b11111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_sat_u_w<const IMM5: i32>(a: v4u32) -> v4u32 {
    static_assert_imm5!(IMM5);
    msa_sat_u_w(a, IMM5)
}

/// Immediate Unsigned Saturate
///
/// Unsigned elements in vector `a` (two unsigned 64-bit integer numbers)
/// are saturated to unsigned values of `imm6+1` bits without changing the data width.
/// The result is stored in the vector (two unsigned 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(sat_u.d, imm1 = 0b111111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_sat_u_d<const IMM6: i32>(a: v2u64) -> v2u64 {
    static_assert_imm6!(IMM6);
    msa_sat_u_d(a, IMM6)
}

/// Immediate Set Shuffle Elements
///
/// The set shuffle instruction works on 4-element sets.
/// All sets are shuffled in the same way: the element i82i+1..2i in `a`
/// (sixteen signed 8-bit integer numbers) is copied over the element i in result vector
/// (sixteen signed 8-bit integer numbers), where i is 0, 1, 2, 3.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(shf.b, imm8 = 0b11111111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_shf_b<const IMM8: i32>(a: v16i8) -> v16i8 {
    static_assert_imm8!(IMM8);
    msa_shf_b(a, IMM8)
}

/// Immediate Set Shuffle Elements
///
/// The set shuffle instruction works on 4-element sets.
/// All sets are shuffled in the same way: the element i82i+1..2i in `a`
/// (eight signed 16-bit integer numbers) is copied over the element i in result vector
/// (eight signed 16-bit integer numbers), where i is 0, 1, 2, 3.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(shf.h, imm8 = 0b11111111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_shf_h<const IMM8: i32>(a: v8i16) -> v8i16 {
    static_assert_imm8!(IMM8);
    msa_shf_h(a, IMM8)
}

/// Immediate Set Shuffle Elements
///
/// The set shuffle instruction works on 4-element sets.
/// All sets are shuffled in the same way: the element i82i+1..2i in `a`
/// (four signed 32-bit integer numbers) is copied over the element i in result vector
/// (four signed 32-bit integer numbers), where i is 0, 1, 2, 3.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(shf.w, imm8 = 0b11111111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_shf_w<const IMM8: i32>(a: v4i32) -> v4i32 {
    static_assert_imm8!(IMM8);
    msa_shf_w(a, IMM8)
}

/// GPR Columns Slide
///
/// Vector registers `a` (sixteen signed 8-bit integer numbers) and `b`
/// (sixteen signed 8-bit integer numbers) contain 2-dimensional byte arrays (rectangles)
/// stored row-wise with as many rows as bytes in integer data format df.
/// The two source rectangles `b` and `a` are concatenated horizontally in the order
/// they appear in the syntax, i.e. first `a` and then `b`. Place a new destination
/// rectangle over `b` and then slide it to the left over the concatenation of `a` and `b`
/// by the number of columns given in GPR `c`.
/// The result is written to vector (sixteen signed 8-bit integer numbers).
/// GPR `c` value is interpreted modulo the number of columns in destination rectangle,
/// or equivalently, the number of data format df elements in the destination vector.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(sld.b))]
pub unsafe fn __msa_sld_b(a: v16i8, b: v16i8, c: i32) -> v16i8 {
    msa_sld_b(a, mem::transmute(b), c)
}

/// GPR Columns Slide
///
/// Vector registers `a` (eight signed 16-bit integer numbers) and `b`
/// (eight signed 16-bit integer numbers) contain 2-dimensional byte arrays (rectangles)
/// stored row-wise with as many rows as bytes in integer data format df.
/// The two source rectangles `b` and `a` are concatenated horizontally in the order
/// they appear in the syntax, i.e. first `a` and then `b`. Place a new destination
/// rectangle over `b` and then slide it to the left over the concatenation of `a` and `b`
/// by the number of columns given in GPR `c`.
/// The result is written to vector (eight signed 16-bit integer numbers).
/// GPR `c` value is interpreted modulo the number of columns in destination rectangle,
/// or equivalently, the number of data format df elements in the destination vector.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(sld.h))]
pub unsafe fn __msa_sld_h(a: v8i16, b: v8i16, c: i32) -> v8i16 {
    msa_sld_h(a, mem::transmute(b), c)
}

/// GPR Columns Slide
///
/// Vector registers `a` (four signed 32-bit integer numbers) and `b`
/// (four signed 32-bit integer numbers) contain 2-dimensional byte arrays (rectangles)
/// stored row-wise with as many rows as bytes in integer data format df.
/// The two source rectangles `b` and `a` are concatenated horizontally in the order
/// they appear in the syntax, i.e. first `a` and then `b`. Place a new destination
/// rectangle over `b` and then slide it to the left over the concatenation of `a` and `b`
/// by the number of columns given in GPR `c`.
/// The result is written to vector (four signed 32-bit integer numbers).
/// GPR `c` value is interpreted modulo the number of columns in destination rectangle,
/// or equivalently, the number of data format df elements in the destination vector.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(sld.w))]
pub unsafe fn __msa_sld_w(a: v4i32, b: v4i32, c: i32) -> v4i32 {
    msa_sld_w(a, mem::transmute(b), c)
}

/// GPR Columns Slide
///
/// Vector registers `a` (two signed 64-bit integer numbers) and `b`
/// (two signed 64-bit integer numbers) contain 2-dimensional byte arrays (rectangles)
/// stored row-wise with as many rows as bytes in integer data format df.
/// The two source rectangles `b` and `a` are concatenated horizontally in the order
/// they appear in the syntax, i.e. first `a` and then `b`. Place a new destination
/// rectangle over `b` and then slide it to the left over the concatenation of `a` and `b`
/// by the number of columns given in GPR `c`.
/// The result is written to vector (two signed 64-bit integer numbers).
/// GPR `c` value is interpreted modulo the number of columns in destination rectangle,
/// or equivalently, the number of data format df elements in the destination vector.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(sld.d))]
pub unsafe fn __msa_sld_d(a: v2i64, b: v2i64, c: i32) -> v2i64 {
    msa_sld_d(a, mem::transmute(b), c)
}

/// Immediate Columns Slide
///
/// Vector registers `a` (sixteen signed 8-bit integer numbers) and `b`
/// (sixteen signed 8-bit integer numbers) contain 2-dimensional byte arrays (rectangles)
/// stored row-wise with as many rows as bytes in integer data format df.
/// The two source rectangles `b` and `a` are concatenated horizontally in the order
/// they appear in the syntax, i.e. first `a` and then `b`. Place a new destination
/// rectangle over `b` and then slide it to the left over the concatenation of `a` and `b`
/// by `imm1` columns.
/// The result is written to vector (sixteen signed 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(sldi.b, imm4 = 0b1111))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn __msa_sldi_b<const IMM4: i32>(a: v16i8, b: v16i8) -> v16i8 {
    static_assert_imm4!(IMM4);
    msa_sldi_b(a, mem::transmute(b), IMM4)
}

/// Immediate Columns Slide
///
/// Vector registers `a` (eight signed 16-bit integer numbers) and `b`
/// (eight signed 16-bit integer numbers) contain 2-dimensional byte arrays (rectangles)
/// stored row-wise with as many rows as bytes in integer data format df.
/// The two source rectangles `b` and `a` are concatenated horizontally in the order
/// they appear in the syntax, i.e. first `a` and then `b`. Place a new destination
/// rectangle over `b` and then slide it to the left over the concatenation of `a` and `b`
/// by `imm1` columns.
/// The result is written to vector (eight signed 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(sldi.h, imm3 = 0b111))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn __msa_sldi_h<const IMM3: i32>(a: v8i16, b: v8i16) -> v8i16 {
    static_assert_imm3!(IMM3);
    msa_sldi_h(a, mem::transmute(b), IMM3)
}

/// Immediate Columns Slide
///
/// Vector registers `a` (four signed 32-bit integer numbers) and `b`
/// (four signed 32-bit integer numbers) contain 2-dimensional byte arrays (rectangles)
/// stored row-wise with as many rows as bytes in integer data format df.
/// The two source rectangles `b` and `a` are concatenated horizontally in the order
/// they appear in the syntax, i.e. first `a` and then `b`. Place a new destination
/// rectangle over `b` and then slide it to the left over the concatenation of `a` and `b`
/// by `imm1` columns.
/// The result is written to vector (four signed 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(sldi.w, imm2 = 0b11))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn __msa_sldi_w<const IMM2: i32>(a: v4i32, b: v4i32) -> v4i32 {
    static_assert_imm2!(IMM2);
    msa_sldi_w(a, mem::transmute(b), IMM2)
}

/// Immediate Columns Slide
///
/// Vector registers `a` (two signed 64-bit integer numbers) and `b`
/// (two signed 64-bit integer numbers) contain 2-dimensional byte arrays (rectangles)
/// stored row-wise with as many rows as bytes in integer data format df.
/// The two source rectangles `b` and `a` are concatenated horizontally in the order
/// they appear in the syntax, i.e. first `a` and then `b`. Place a new destination
/// rectangle over `b` and then slide it to the left over the concatenation of `a` and `b`
/// by `imm1` columns.
/// The result is written to vector (two signed 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(sldi.d, imm1 = 0b1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn __msa_sldi_d<const IMM1: i32>(a: v2i64, b: v2i64) -> v2i64 {
    static_assert_imm1!(IMM1);
    msa_sldi_d(a, mem::transmute(b), IMM1)
}

/// Vector Shift Left
///
/// The elements in vector `a` (sixteen signed 8-bit integer numbers)
/// are shifted left by the number of bits the elements in vector `b`
/// (sixteen signed 8-bit integer numbers) specify modulo the size of the
/// element in bits. The result is written to vector (sixteen signed 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(sll.b))]
pub unsafe fn __msa_sll_b(a: v16i8, b: v16i8) -> v16i8 {
    msa_sll_b(a, mem::transmute(b))
}

/// Vector Shift Left
///
/// The elements in vector `a` (eight signed 16-bit integer numbers)
/// are shifted left by the number of bits the elements in vector `b`
/// (eight signed 16-bit integer numbers) specify modulo the size of the
/// element in bits. The result is written to vector (eight signed 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(sll.h))]
pub unsafe fn __msa_sll_h(a: v8i16, b: v8i16) -> v8i16 {
    msa_sll_h(a, mem::transmute(b))
}

/// Vector Shift Left
///
/// The elements in vector `a` (four signed 32-bit integer numbers)
/// are shifted left by the number of bits the elements in vector `b`
/// (four signed 32-bit integer numbers) specify modulo the size of the
/// element in bits. The result is written to vector (four signed 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(sll.w))]
pub unsafe fn __msa_sll_w(a: v4i32, b: v4i32) -> v4i32 {
    msa_sll_w(a, mem::transmute(b))
}

/// Vector Shift Left
///
/// The elements in vector `a` (two signed 64-bit integer numbers)
/// are shifted left by the number of bits the elements in vector `b`
/// (two signed 64-bit integer numbers) specify modulo the size of the
/// element in bits. The result is written to vector (two signed 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(sll.d))]
pub unsafe fn __msa_sll_d(a: v2i64, b: v2i64) -> v2i64 {
    msa_sll_d(a, mem::transmute(b))
}

/// Immediate Shift Left
///
/// The elements in vector `a` (sixteen signed 8-bit integer numbers)
/// are shifted left by `imm4` bits.
/// The result is written to vector (sixteen signed 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(slli.b, imm4 = 0b1111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_slli_b<const IMM4: i32>(a: v16i8) -> v16i8 {
    static_assert_imm4!(IMM4);
    msa_slli_b(a, IMM4)
}

/// Immediate Shift Left
///
/// The elements in vector `a` (eight signed 16-bit integer numbers)
/// are shifted left by `imm3` bits.
/// The result is written to vector (eight signed 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(slli.h, imm3 = 0b111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_slli_h<const IMM3: i32>(a: v8i16) -> v8i16 {
    static_assert_imm3!(IMM3);
    msa_slli_h(a, IMM3)
}

/// Immediate Shift Left
///
/// The elements in vector `a` (four signed 32-bit integer numbers)
/// are shifted left by `imm2` bits.
/// The result is written to vector (four signed 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(slli.w, imm2 = 0b11))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_slli_w<const IMM2: i32>(a: v4i32) -> v4i32 {
    static_assert_imm2!(IMM2);
    msa_slli_w(a, IMM2)
}

/// Immediate Shift Left
///
/// The elements in vector `a` (two signed 64-bit integer numbers)
/// are shifted left by `imm1` bits.
/// The result is written to vector (two signed 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(slli.d, imm1 = 0b1))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_slli_d<const IMM1: i32>(a: v2i64) -> v2i64 {
    static_assert_imm1!(IMM1);
    msa_slli_d(a, IMM1)
}

/// GPR Element Splat
///
/// Replicate vector `a` (sixteen signed 8-bit integer numbers)
/// element with index given by GPR `b` to all elements in vector
/// (sixteen signed 8-bit integer numbers) GPR `b` value is interpreted
/// modulo the number of data format df elements in the destination vector.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(splat.b))]
pub unsafe fn __msa_splat_b(a: v16i8, b: i32) -> v16i8 {
    msa_splat_b(a, mem::transmute(b))
}

/// GPR Element Splat
///
/// Replicate vector `a` (eight signed 16-bit integer numbers)
/// element with index given by GPR `b` to all elements in vector
/// (eight signed 16-bit integer numbers) GPR `b` value is interpreted
/// modulo the number of data format df elements in the destination vector.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(splat.h))]
pub unsafe fn __msa_splat_h(a: v8i16, b: i32) -> v8i16 {
    msa_splat_h(a, mem::transmute(b))
}

/// GPR Element Splat
///
/// Replicate vector `a` (four signed 32-bit integer numbers)
/// element with index given by GPR `b` to all elements in vector
/// (four signed 32-bit integer numbers) GPR `b` value is interpreted
/// modulo the number of data format df elements in the destination vector.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(splat.w))]
pub unsafe fn __msa_splat_w(a: v4i32, b: i32) -> v4i32 {
    msa_splat_w(a, mem::transmute(b))
}

/// GPR Element Splat
///
/// Replicate vector `a` (two signed 64-bit integer numbers)
/// element with index given by GPR `b` to all elements in vector
/// (two signed 64-bit integer numbers) GPR `b` value is interpreted
/// modulo the number of data format df elements in the destination vector.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(splat.d))]
pub unsafe fn __msa_splat_d(a: v2i64, b: i32) -> v2i64 {
    msa_splat_d(a, mem::transmute(b))
}

/// Immediate Element Splat
///
/// Replicate element `imm4` in vector `a` (sixteen signed 8-bit integer numbers)
/// to all elements in vector (sixteen signed 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(splati.b, imm4 = 0b1111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_splati_b<const IMM4: i32>(a: v16i8) -> v16i8 {
    static_assert_imm4!(IMM4);
    msa_splati_b(a, IMM4)
}

/// Immediate Element Splat
///
/// Replicate element `imm3` in vector `a` (eight signed 16-bit integer numbers)
/// to all elements in vector (eight signed 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(splati.h, imm3 = 0b111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_splati_h<const IMM3: i32>(a: v8i16) -> v8i16 {
    static_assert_imm3!(IMM3);
    msa_splati_h(a, IMM3)
}

/// Immediate Element Splat
///
/// Replicate element `imm2` in vector `a` (four signed 32-bit integer numbers)
/// to all elements in vector (four signed 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(splati.w, imm2 = 0b11))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_splati_w<const IMM2: i32>(a: v4i32) -> v4i32 {
    static_assert_imm2!(IMM2);
    msa_splati_w(a, IMM2)
}

/// Immediate Element Splat
///
/// Replicate element `imm1` in vector `a` (two signed 64-bit integer numbers)
/// to all elements in vector (two signed 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(splati.d, imm1 = 0b1))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_splati_d<const IMM1: i32>(a: v2i64) -> v2i64 {
    static_assert_imm1!(IMM1);
    msa_splati_d(a, IMM1)
}

/// Vector Shift Right Arithmetic
///
/// The elements in vector `a` (sixteen signed 8-bit integer numbers)
/// are shifted right arithmetic by the number of bits the elements in vector `b`
/// (sixteen signed 8-bit integer numbers) specify modulo the size of the
/// element in bits.The result is written to vector (sixteen signed 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(sra.b))]
pub unsafe fn __msa_sra_b(a: v16i8, b: v16i8) -> v16i8 {
    msa_sra_b(a, mem::transmute(b))
}

/// Vector Shift Right Arithmetic
///
/// The elements in vector `a` (eight signed 16-bit integer numbers)
/// are shifted right arithmetic by the number of bits the elements in vector `b`
/// (eight signed 16-bit integer numbers) specify modulo the size of the
/// element in bits.The result is written to vector (eight signed 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(sra.h))]
pub unsafe fn __msa_sra_h(a: v8i16, b: v8i16) -> v8i16 {
    msa_sra_h(a, mem::transmute(b))
}

/// Vector Shift Right Arithmetic
///
/// The elements in vector `a` (four signed 32-bit integer numbers)
/// are shifted right arithmetic by the number of bits the elements in vector `b`
/// (four signed 32-bit integer numbers) specify modulo the size of the
/// element in bits.The result is written to vector (four signed 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(sra.w))]
pub unsafe fn __msa_sra_w(a: v4i32, b: v4i32) -> v4i32 {
    msa_sra_w(a, mem::transmute(b))
}

/// Vector Shift Right Arithmetic
///
/// The elements in vector `a` (two signed 64-bit integer numbers)
/// are shifted right arithmetic by the number of bits the elements in vector `b`
/// (two signed 64-bit integer numbers) specify modulo the size of the
/// element in bits.The result is written to vector (two signed 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(sra.d))]
pub unsafe fn __msa_sra_d(a: v2i64, b: v2i64) -> v2i64 {
    msa_sra_d(a, mem::transmute(b))
}

/// Immediate Shift Right Arithmetic
///
/// The elements in vector `a` (sixteen signed 8-bit integer numbers)
/// are shifted right arithmetic by `imm3` bits.
/// The result is written to vector (sixteen signed 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(srai.b, imm3 = 0b111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_srai_b<const IMM3: i32>(a: v16i8) -> v16i8 {
    static_assert_imm3!(IMM3);
    msa_srai_b(a, IMM3)
}

/// Immediate Shift Right Arithmetic
///
/// The elements in vector `a` (eight signed 16-bit integer numbers)
/// are shifted right arithmetic by `imm4` bits.
/// The result is written to vector (eight signed 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(srai.h, imm4 = 0b1111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_srai_h<const IMM4: i32>(a: v8i16) -> v8i16 {
    static_assert_imm4!(IMM4);
    msa_srai_h(a, IMM4)
}

/// Immediate Shift Right Arithmetic
///
/// The elements in vector `a` (four signed 32-bit integer numbers)
/// are shifted right arithmetic by `imm5` bits.
/// The result is written to vector (four signed 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(srai.w, imm5 = 0b11111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_srai_w<const IMM5: i32>(a: v4i32) -> v4i32 {
    static_assert_imm5!(IMM5);
    msa_srai_w(a, IMM5)
}

/// Immediate Shift Right Arithmetic
///
/// The elements in vector `a` (two signed 64-bit integer numbers)
/// are shifted right arithmetic by `imm6` bits.
/// The result is written to vector (two signed 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(srai.d, imm6 = 0b111111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_srai_d<const IMM6: i32>(a: v2i64) -> v2i64 {
    static_assert_imm6!(IMM6);
    msa_srai_d(a, IMM6)
}

/// Vector Shift Right Arithmetic Rounded
///
/// The elements in vector `a` (sixteen signed 8-bit integer numbers)
/// are shifted right arithmetic by the number of bits the elements in vector `b`
/// (sixteen signed 8-bit integer numbers) specify modulo the size of the
/// element in bits.The most significant discarded bit is added to the shifted
/// value (for rounding) and the result is written to vector (sixteen signed 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(srar.b))]
pub unsafe fn __msa_srar_b(a: v16i8, b: v16i8) -> v16i8 {
    msa_srar_b(a, mem::transmute(b))
}

/// Vector Shift Right Arithmetic Rounded
///
/// The elements in vector `a` (eight signed 16-bit integer numbers)
/// are shifted right arithmetic by the number of bits the elements in vector `b`
/// (eight signed 16-bit integer numbers) specify modulo the size of the
/// element in bits.The most significant discarded bit is added to the shifted
/// value (for rounding) and the result is written to vector (eight signed 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(srar.h))]
pub unsafe fn __msa_srar_h(a: v8i16, b: v8i16) -> v8i16 {
    msa_srar_h(a, mem::transmute(b))
}

/// Vector Shift Right Arithmetic Rounded
///
/// The elements in vector `a` (four signed 32-bit integer numbers)
/// are shifted right arithmetic by the number of bits the elements in vector `b`
/// (four signed 32-bit integer numbers) specify modulo the size of the
/// element in bits.The most significant discarded bit is added to the shifted
/// value (for rounding) and the result is written to vector (four signed 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(srar.w))]
pub unsafe fn __msa_srar_w(a: v4i32, b: v4i32) -> v4i32 {
    msa_srar_w(a, mem::transmute(b))
}

/// Vector Shift Right Arithmetic Rounded
///
/// The elements in vector `a` (two signed 64-bit integer numbers)
/// are shifted right arithmetic by the number of bits the elements in vector `b`
/// (two signed 64-bit integer numbers) specify modulo the size of the
/// element in bits.The most significant discarded bit is added to the shifted
/// value (for rounding) and the result is written to vector (two signed 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(srar.d))]
pub unsafe fn __msa_srar_d(a: v2i64, b: v2i64) -> v2i64 {
    msa_srar_d(a, mem::transmute(b))
}

/// Immediate Shift Right Arithmetic Rounded
///
/// The elements in vector `a` (sixteen signed 8-bit integer numbers)
/// are shifted right arithmetic by `imm3` bits.The most significant
/// discarded bit is added to the shifted value (for rounding) and
/// the result is written to vector (sixteen signed 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(srari.b, imm3 = 0b111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_srari_b<const IMM3: i32>(a: v16i8) -> v16i8 {
    static_assert_imm3!(IMM3);
    msa_srari_b(a, IMM3)
}

/// Immediate Shift Right Arithmetic Rounded
///
/// The elements in vector `a` (eight signed 16-bit integer numbers)
/// are shifted right arithmetic by `imm4` bits.The most significant
/// discarded bit is added to the shifted value (for rounding) and
/// the result is written to vector (eight signed 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(srari.h, imm4 = 0b1111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_srari_h<const IMM4: i32>(a: v8i16) -> v8i16 {
    static_assert_imm4!(IMM4);
    msa_srari_h(a, IMM4)
}

/// Immediate Shift Right Arithmetic Rounded
///
/// The elements in vector `a` (four signed 32-bit integer numbers)
/// are shifted right arithmetic by `imm5` bits.The most significant
/// discarded bit is added to the shifted value (for rounding) and
/// the result is written to vector (four signed 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(srari.w, imm5 = 0b11111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_srari_w<const IMM5: i32>(a: v4i32) -> v4i32 {
    static_assert_imm5!(IMM5);
    msa_srari_w(a, IMM5)
}

/// Immediate Shift Right Arithmetic Rounded
///
/// The elements in vector `a` (two signed 64-bit integer numbers)
/// are shifted right arithmetic by `imm6` bits.The most significant
/// discarded bit is added to the shifted value (for rounding) and
/// the result is written to vector (two signed 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(srari.d, imm6 = 0b111111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_srari_d<const IMM6: i32>(a: v2i64) -> v2i64 {
    static_assert_imm6!(IMM6);
    msa_srari_d(a, IMM6)
}

/// Vector Shift Right Logical
///
/// The elements in vector `a` (sixteen signed 8-bit integer numbers)
/// are shifted right logical by the number of bits the elements in vector `b`
/// (sixteen signed 8-bit integer numbers) specify modulo the size of the
/// element in bits.The result is written to vector (sixteen signed 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(srl.b))]
pub unsafe fn __msa_srl_b(a: v16i8, b: v16i8) -> v16i8 {
    msa_srl_b(a, mem::transmute(b))
}

/// Vector Shift Right Logical
///
/// The elements in vector `a` (eight signed 16-bit integer numbers)
/// are shifted right logical by the number of bits the elements in vector `b`
/// (eight signed 16-bit integer numbers) specify modulo the size of the
/// element in bits.The result is written to vector (eight signed 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(srl.h))]
pub unsafe fn __msa_srl_h(a: v8i16, b: v8i16) -> v8i16 {
    msa_srl_h(a, mem::transmute(b))
}

/// Vector Shift Right Logical
///
/// The elements in vector `a` (four signed 32-bit integer numbers)
/// are shifted right logical by the number of bits the elements in vector `b`
/// (four signed 32-bit integer numbers) specify modulo the size of the
/// element in bits.The result is written to vector (four signed 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(srl.w))]
pub unsafe fn __msa_srl_w(a: v4i32, b: v4i32) -> v4i32 {
    msa_srl_w(a, mem::transmute(b))
}

/// Vector Shift Right Logical
///
/// The elements in vector `a` (two signed 64-bit integer numbers)
/// are shifted right logical by the number of bits the elements in vector `b`
/// (two signed 64-bit integer numbers) specify modulo the size of the
/// element in bits.The result is written to vector (two signed 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(srl.d))]
pub unsafe fn __msa_srl_d(a: v2i64, b: v2i64) -> v2i64 {
    msa_srl_d(a, mem::transmute(b))
}

/// Immediate Shift Right Logical
///
/// The elements in vector `a` (sixteen signed 8-bit integer numbers)
/// are shifted right logical by `imm4` bits.
/// The result is written to vector (sixteen signed 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(srli.b, imm4 = 0b1111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_srli_b<const IMM4: i32>(a: v16i8) -> v16i8 {
    static_assert_imm4!(IMM4);
    msa_srli_b(a, IMM4)
}

/// Immediate Shift Right Logical
///
/// The elements in vector `a` (eight signed 16-bit integer numbers)
/// are shifted right logical by `imm3` bits.
/// The result is written to vector (eight signed 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(srli.h, imm3 = 0b111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_srli_h<const IMM3: i32>(a: v8i16) -> v8i16 {
    static_assert_imm3!(IMM3);
    msa_srli_h(a, IMM3)
}

/// Immediate Shift Right Logical
///
/// The elements in vector `a` (four signed 32-bit integer numbers)
/// are shifted right logical by `imm2` bits.
/// The result is written to vector (four signed 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(srli.w, imm2 = 0b11))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_srli_w<const IMM2: i32>(a: v4i32) -> v4i32 {
    static_assert_imm2!(IMM2);
    msa_srli_w(a, IMM2)
}

/// Immediate Shift Right Logical
///
/// The elements in vector `a` (two signed 64-bit integer numbers)
/// are shifted right logical by `imm1` bits.
/// The result is written to vector (two signed 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(srli.d, imm1 = 0b1))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_srli_d<const IMM1: i32>(a: v2i64) -> v2i64 {
    static_assert_imm1!(IMM1);
    msa_srli_d(a, IMM1)
}

/// Vector Shift Right Logical Rounded
///
/// The elements in vector `a` (sixteen signed 8-bit integer numbers)
/// are shifted right logical by the number of bits the elements in vector `b`
/// (sixteen signed 8-bit integer numbers) specify modulo the size of the
/// element in bits.The most significant discarded bit is added to the shifted
/// value (for rounding) and the result is written to vector (sixteen signed 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(srlr.b))]
pub unsafe fn __msa_srlr_b(a: v16i8, b: v16i8) -> v16i8 {
    msa_srlr_b(a, mem::transmute(b))
}

/// Vector Shift Right Logical Rounded
///
/// The elements in vector `a` (eight signed 16-bit integer numbers)
/// are shifted right logical by the number of bits the elements in vector `b`
/// (eight signed 16-bit integer numbers) specify modulo the size of the
/// element in bits.The most significant discarded bit is added to the shifted
/// value (for rounding) and the result is written to vector (eight signed 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(srlr.h))]
pub unsafe fn __msa_srlr_h(a: v8i16, b: v8i16) -> v8i16 {
    msa_srlr_h(a, mem::transmute(b))
}

/// Vector Shift Right Logical Rounded
///
/// The elements in vector `a` (four signed 32-bit integer numbers)
/// are shifted right logical by the number of bits the elements in vector `b`
/// (four signed 32-bit integer numbers) specify modulo the size of the
/// element in bits.The most significant discarded bit is added to the shifted
/// value (for rounding) and the result is written to vector (four signed 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(srlr.w))]
pub unsafe fn __msa_srlr_w(a: v4i32, b: v4i32) -> v4i32 {
    msa_srlr_w(a, mem::transmute(b))
}

/// Vector Shift Right Logical Rounded
///
/// The elements in vector `a` (two signed 64-bit integer numbers)
/// are shifted right logical by the number of bits the elements in vector `b`
/// (two signed 64-bit integer numbers) specify modulo the size of the
/// element in bits.The most significant discarded bit is added to the shifted
/// value (for rounding) and the result is written to vector (two signed 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(srlr.d))]
pub unsafe fn __msa_srlr_d(a: v2i64, b: v2i64) -> v2i64 {
    msa_srlr_d(a, mem::transmute(b))
}

/// Immediate Shift Right Logical Rounded
///
/// The elements in vector `a` (sixteen signed 8-bit integer numbers)
/// are shifted right logical by `imm6` bits.The most significant
/// discarded bit is added to the shifted value (for rounding) and
/// the result is written to vector (sixteen signed 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(srlri.b, imm3 = 0b111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_srlri_b<const IMM3: i32>(a: v16i8) -> v16i8 {
    static_assert_imm3!(IMM3);
    msa_srlri_b(a, IMM3)
}

/// Immediate Shift Right Logical Rounded
///
/// The elements in vector `a` (eight signed 16-bit integer numbers)
/// are shifted right logical by `imm6` bits.The most significant
/// discarded bit is added to the shifted value (for rounding) and
/// the result is written to vector (eight signed 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(srlri.h, imm4 = 0b1111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_srlri_h<const IMM4: i32>(a: v8i16) -> v8i16 {
    static_assert_imm4!(IMM4);
    msa_srlri_h(a, IMM4)
}

/// Immediate Shift Right Logical Rounded
///
/// The elements in vector `a` (four signed 32-bit integer numbers)
/// are shifted right logical by `imm6` bits.The most significant
/// discarded bit is added to the shifted value (for rounding) and
/// the result is written to vector (four signed 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(srlri.w, imm5 = 0b11111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_srlri_w<const IMM5: i32>(a: v4i32) -> v4i32 {
    static_assert_imm5!(IMM5);
    msa_srlri_w(a, IMM5)
}

/// Immediate Shift Right Logical Rounded
///
/// The elements in vector `a` (two signed 64-bit integer numbers)
/// are shifted right logical by `imm6` bits.The most significant
/// discarded bit is added to the shifted value (for rounding) and
/// the result is written to vector (two signed 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(srlri.d, imm6 = 0b111111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_srlri_d<const IMM6: i32>(a: v2i64) -> v2i64 {
    static_assert_imm6!(IMM6);
    msa_srlri_d(a, IMM6)
}

/// Vector Store
///
/// The WRLEN / 8 bytes in vector `a` (sixteen signed 8-bit integer numbers)
/// are stored as elements of data format df at the effective memory location
/// addressed by the base `mem_addr` and the 10-bit signed immediate offset `imm_s10`.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(st.b, imm_s10 = 0b1111111111))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn __msa_st_b<const IMM_S10: i32>(a: v16i8, mem_addr: *mut u8) -> () {
    static_assert_imm_s10!(IMM_S10);
    msa_st_b(a, mem_addr, IMM_S10)
}

/// Vector Store
///
/// The WRLEN / 8 bytes in vector `a` (eight signed 16-bit integer numbers)
/// are stored as elements of data format df at the effective memory location
/// addressed by the base `mem_addr` and the 11-bit signed immediate offset `imm_s11`.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(st.h, imm_s11 = 0b11111111111))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn __msa_st_h<const IMM_S11: i32>(a: v8i16, mem_addr: *mut u8) -> () {
    static_assert_imm_s11!(IMM_S11);
    static_assert!(IMM_S11: i32 where IMM_S11 % 2 == 0);
    msa_st_h(a, mem_addr, IMM_S11)
}

/// Vector Store
///
/// The WRLEN / 8 bytes in vector `a` (four signed 32-bit integer numbers)
/// are stored as elements of data format df at the effective memory location
/// addressed by the base `mem_addr` and the 12-bit signed immediate offset `imm_s12`.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(st.w, imm_s12 = 0b111111111111))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn __msa_st_w<const IMM_S12: i32>(a: v4i32, mem_addr: *mut u8) -> () {
    static_assert_imm_s12!(IMM_S12);
    static_assert!(IMM_S12: i32 where IMM_S12 % 4 == 0);
    msa_st_w(a, mem_addr, IMM_S12)
}

/// Vector Store
///
/// The WRLEN / 8 bytes in vector `a` (two signed 64-bit integer numbers)
/// are stored as elements of data format df at the effective memory location
/// addressed by the base `mem_addr` and the 13-bit signed immediate offset `imm_s13`.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(st.d, imm_s13 = 0b1111111111111))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn __msa_st_d<const IMM_S13: i32>(a: v2i64, mem_addr: *mut u8) -> () {
    static_assert_imm_s13!(IMM_S13);
    static_assert!(IMM_S13: i32 where IMM_S13 % 8 == 0);
    msa_st_d(a, mem_addr, IMM_S13)
}

/// Vector Signed Saturated Subtract of Signed Values
///
/// The elements in vector `b` (sixteen signed 8-bit integer numbers)
/// are subtracted from the elements in vector `a` (sixteen signed 8-bit integer numbers).
/// Signed arithmetic is performed and overflows clamp to the largest and/or smallest
/// representable signed values before writing the result to vector (sixteen signed 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(subs_s.b))]
pub unsafe fn __msa_subs_s_b(a: v16i8, b: v16i8) -> v16i8 {
    msa_subs_s_b(a, mem::transmute(b))
}

/// Vector Signed Saturated Subtract of Signed Values
///
/// The elements in vector `b` (eight signed 16-bit integer numbers)
/// are subtracted from the elements in vector `a` (eight signed 16-bit integer numbers).
/// Signed arithmetic is performed and overflows clamp to the largest and/or smallest
/// representable signed values before writing the result to vector (eight signed 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(subs_s.h))]
pub unsafe fn __msa_subs_s_h(a: v8i16, b: v8i16) -> v8i16 {
    msa_subs_s_h(a, mem::transmute(b))
}

/// Vector Signed Saturated Subtract of Signed Values
///
/// The elements in vector `b` (four signed 32-bit integer numbers)
/// are subtracted from the elements in vector `a` (four signed 32-bit integer numbers).
/// Signed arithmetic is performed and overflows clamp to the largest and/or smallest
/// representable signed values before writing the result to vector (four signed 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(subs_s.w))]
pub unsafe fn __msa_subs_s_w(a: v4i32, b: v4i32) -> v4i32 {
    msa_subs_s_w(a, mem::transmute(b))
}

/// Vector Signed Saturated Subtract of Signed Values
///
/// The elements in vector `b` (two signed 64-bit integer numbers)
/// are subtracted from the elements in vector `a` (two signed 64-bit integer numbers).
/// Signed arithmetic is performed and overflows clamp to the largest and/or smallest
/// representable signed values before writing the result to vector (two signed 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(subs_s.d))]
pub unsafe fn __msa_subs_s_d(a: v2i64, b: v2i64) -> v2i64 {
    msa_subs_s_d(a, mem::transmute(b))
}

/// Vector Unsigned Saturated Subtract of Unsigned Values
///
/// The elements in vector `b` (sixteen unsigned 8-bit integer numbers)
/// are subtracted from the elements in vector `a` (sixteen unsigned 8-bit integer numbers).
/// Unsigned arithmetic is performed and under-flows clamp to 0 before writing
/// the result to vector (sixteen unsigned 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(subs_u.b))]
pub unsafe fn __msa_subs_u_b(a: v16u8, b: v16u8) -> v16u8 {
    msa_subs_u_b(a, mem::transmute(b))
}

/// Vector Unsigned Saturated Subtract of Unsigned Values
///
/// The elements in vector `b` (eight unsigned 16-bit integer numbers)
/// are subtracted from the elements in vector `a` (eight unsigned 16-bit integer numbers).
/// Unsigned arithmetic is performed and under-flows clamp to 0 before writing
/// the result to vector (eight unsigned 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(subs_u.h))]
pub unsafe fn __msa_subs_u_h(a: v8u16, b: v8u16) -> v8u16 {
    msa_subs_u_h(a, mem::transmute(b))
}

/// Vector Unsigned Saturated Subtract of Unsigned Values
///
/// The elements in vector `b` (four unsigned 32-bit integer numbers)
/// are subtracted from the elements in vector `a` (four unsigned 32-bit integer numbers).
/// Unsigned arithmetic is performed and under-flows clamp to 0 before writing
/// the result to vector (four unsigned 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(subs_u.w))]
pub unsafe fn __msa_subs_u_w(a: v4u32, b: v4u32) -> v4u32 {
    msa_subs_u_w(a, mem::transmute(b))
}

/// Vector Unsigned Saturated Subtract of Unsigned Values
///
/// The elements in vector `b` (two unsigned 64-bit integer numbers)
/// are subtracted from the elements in vector `a` (two unsigned 64-bit integer numbers).
/// Unsigned arithmetic is performed and under-flows clamp to 0 before writing
/// the result to vector (two unsigned 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(subs_u.d))]
pub unsafe fn __msa_subs_u_d(a: v2u64, b: v2u64) -> v2u64 {
    msa_subs_u_d(a, mem::transmute(b))
}

/// Vector Unsigned Saturated Subtract of Signed from Unsigned
///
/// The signed elements in vector `b` (sixteen signed 8-bit integer numbers)
/// are subtracted from the unsigned elements in vector `a` (sixteen unsigned 8-bit integer numbers).
/// The signed result is unsigned saturated and written to
/// to vector (sixteen unsigned 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(subsus_u.b))]
pub unsafe fn __msa_subsus_u_b(a: v16u8, b: v16i8) -> v16u8 {
    msa_subsus_u_b(a, mem::transmute(b))
}

/// Vector Unsigned Saturated Subtract of Signed from Unsigned
///
/// The signed elements in vector `b` (eight signed 16-bit integer numbers)
/// are subtracted from the unsigned elements in vector `a` (eight unsigned 16-bit integer numbers).
/// The signed result is unsigned saturated and written to
/// to vector (eight unsigned 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(subsus_u.h))]
pub unsafe fn __msa_subsus_u_h(a: v8u16, b: v8i16) -> v8u16 {
    msa_subsus_u_h(a, mem::transmute(b))
}

/// Vector Unsigned Saturated Subtract of Signed from Unsigned
///
/// The signed elements in vector `b` (four signed 6432it integer numbers)
/// are subtracted from the unsigned elements in vector `a` (four unsigned 32-bit integer numbers).
/// The signed result is unsigned saturated and written to
/// to vector (four unsigned 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(subsus_u.w))]
pub unsafe fn __msa_subsus_u_w(a: v4u32, b: v4i32) -> v4u32 {
    msa_subsus_u_w(a, mem::transmute(b))
}

/// Vector Unsigned Saturated Subtract of Signed from Unsigned
///
/// The signed elements in vector `b` (two signed 64-bit integer numbers)
/// are subtracted from the unsigned elements in vector `a` (two unsigned 64-bit integer numbers).
/// The signed result is unsigned saturated and written to
/// to vector (two unsigned 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(subsus_u.d))]
pub unsafe fn __msa_subsus_u_d(a: v2u64, b: v2i64) -> v2u64 {
    msa_subsus_u_d(a, mem::transmute(b))
}

/// Vector Signed Saturated Subtract of Unsigned Values
///
/// The unsigned elements in vector `b` (sixteen unsigned 8-bit integer numbers)
/// are subtracted from the unsigned elements in vector `a` (sixteen unsigned 8-bit integer numbers).
/// The signed result is signed saturated and written to
/// to vector (sixteen unsigned 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(subsuu_s.b))]
pub unsafe fn __msa_subsuu_s_b(a: v16u8, b: v16u8) -> v16i8 {
    msa_subsuu_s_b(a, mem::transmute(b))
}

/// Vector Signed Saturated Subtract of Unsigned Values
///
/// The unsigned elements in vector `b` (eight unsigned 16-bit integer numbers)
/// are subtracted from the unsigned elements in vector `a` (eight unsigned 16-bit integer numbers).
/// The signed result is signed saturated and written to
/// to vector (eight unsigned 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(subsuu_s.h))]
pub unsafe fn __msa_subsuu_s_h(a: v8u16, b: v8u16) -> v8i16 {
    msa_subsuu_s_h(a, mem::transmute(b))
}

/// Vector Signed Saturated Subtract of Unsigned Values
///
/// The unsigned elements in vector `b` (four unsigned 32-bit integer numbers)
/// are subtracted from the unsigned elements in vector `a` (four unsigned 32-bit integer numbers).
/// The signed result is signed saturated and written to
/// to vector (four unsigned 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(subsuu_s.w))]
pub unsafe fn __msa_subsuu_s_w(a: v4u32, b: v4u32) -> v4i32 {
    msa_subsuu_s_w(a, mem::transmute(b))
}

/// Vector Signed Saturated Subtract of Unsigned Values
///
/// The unsigned elements in vector `b` (two unsigned 64-bit integer numbers)
/// are subtracted from the unsigned elements in vector `a` (two unsigned 64-bit integer numbers).
/// The signed result is signed saturated and written to
/// to vector (two unsigned 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(subsuu_s.d))]
pub unsafe fn __msa_subsuu_s_d(a: v2u64, b: v2u64) -> v2i64 {
    msa_subsuu_s_d(a, mem::transmute(b))
}

/// Vector Subtract
///
/// The elements in vector `b` (sixteen signed 8-bit integer numbers)
/// are subtracted from the elements in vector `a` (sixteen signed 8-bit integer numbers).
/// The result is written to vector (sixteen signed 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(subv.b))]
pub unsafe fn __msa_subv_b(a: v16i8, b: v16i8) -> v16i8 {
    msa_subv_b(a, mem::transmute(b))
}

/// Vector Subtract
///
/// The elements in vector `b` (eight signed 16-bit integer numbers)
/// are subtracted from the elements in vector `a` (eight signed 16-bit integer numbers).
/// The result is written to vector (eight signed 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(subv.h))]
pub unsafe fn __msa_subv_h(a: v8i16, b: v8i16) -> v8i16 {
    msa_subv_h(a, mem::transmute(b))
}

/// Vector Subtract
///
/// The elements in vector `b` (four signed 32-bit integer numbers)
/// are subtracted from the elements in vector `a` (four signed 32-bit integer numbers).
/// The result is written to vector (four signed 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(subv.w))]
pub unsafe fn __msa_subv_w(a: v4i32, b: v4i32) -> v4i32 {
    msa_subv_w(a, mem::transmute(b))
}

/// Vector Subtract
///
/// The elements in vector `b` (two signed 64-bit integer numbers)
/// are subtracted from the elements in vector `a` (two signed 64-bit integer numbers).
/// The result is written to vector (two signed 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(subv.d))]
pub unsafe fn __msa_subv_d(a: v2i64, b: v2i64) -> v2i64 {
    msa_subv_d(a, mem::transmute(b))
}

/// Immediate Subtract
///
/// The 5-bit immediate unsigned value `imm5`
/// are subtracted from the elements in vector `a` (sixteen signed 8-bit integer numbers).
/// The result is written to vector (sixteen signed 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(subvi.b, imm5 = 0b10111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_subvi_b<const IMM5: i32>(a: v16i8) -> v16i8 {
    static_assert_imm5!(IMM5);
    msa_subvi_b(a, IMM5)
}

/// Immediate Subtract
///
/// The 5-bit immediate unsigned value `imm5`
/// are subtracted from the elements in vector `a` (eight signed 16-bit integer numbers).
/// The result is written to vector (eight signed 16-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(subvi.h, imm5 = 0b10111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_subvi_h<const IMM5: i32>(a: v8i16) -> v8i16 {
    static_assert_imm5!(IMM5);
    msa_subvi_h(a, IMM5)
}

/// Immediate Subtract
///
/// The 5-bit immediate unsigned value `imm5`
/// are subtracted from the elements in vector `a` (four signed 32-bit integer numbers).
/// The result is written to vector (four signed 32-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(subvi.w, imm5 = 0b10111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_subvi_w<const IMM5: i32>(a: v4i32) -> v4i32 {
    static_assert_imm5!(IMM5);
    msa_subvi_w(a, IMM5)
}

/// Immediate Subtract
///
/// The 5-bit immediate unsigned value `imm5`
/// are subtracted from the elements in vector `a` (two signed 64-bit integer numbers).
/// The result is written to vector (two signed 64-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(subvi.d, imm5 = 0b10111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_subvi_d<const IMM5: i32>(a: v2i64) -> v2i64 {
    static_assert_imm5!(IMM5);
    msa_subvi_d(a, IMM5)
}

/// Vector Data Preserving Shuffle
///
/// The vector shuffle instructions selectively copy data elements from the
/// concatenation of vectors `b` (sixteen signed 8-bit integer numbers)
/// and `c` (sixteen signed 8-bit integer numbers) in to vector `a`
/// (sixteen signed 8-bit integer numbers) based on the corresponding control element in `a`.
/// The least significant 6 bits in `a` control elements modulo the number of elements in
/// the concatenated vectors `b`, `a` specify the index of the source element.
/// If bit 6 or bit 7 is 1, there will be no copy, but rather the destination element is set to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(vshf.b))]
pub unsafe fn __msa_vshf_b(a: v16i8, b: v16i8, c: v16i8) -> v16i8 {
    msa_vshf_b(a, mem::transmute(b), c)
}

/// Vector Data Preserving Shuffle
///
/// The vector shuffle instructions selectively copy data elements from the
/// concatenation of vectors `b` (eight signed 16-bit integer numbers)
/// and `c` (eight signed 16-bit integer numbers) in to vector `a`
/// (eight signed 16-bit integer numbers) based on the corresponding control element in `a`.
/// The least significant 6 bits in `a` control elements modulo the number of elements in
/// the concatenated vectors `b`, `a` specify the index of the source element.
/// If bit 6 or bit 7 is 1, there will be no copy, but rather the destination element is set to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(vshf.h))]
pub unsafe fn __msa_vshf_h(a: v8i16, b: v8i16, c: v8i16) -> v8i16 {
    msa_vshf_h(a, mem::transmute(b), c)
}

/// Vector Data Preserving Shuffle
///
/// The vector shuffle instructions selectively copy data elements from the
/// concatenation of vectors `b` (four signed 32-bit integer numbers)
/// and `c` (four signed 32-bit integer numbers) in to vector `a`
/// (four signed 32-bit integer numbers) based on the corresponding control element in `a`.
/// The least significant 6 bits in `a` control elements modulo the number of elements in
/// the concatenated vectors `b`, `a` specify the index of the source element.
/// If bit 6 or bit 7 is 1, there will be no copy, but rather the destination element is set to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(vshf.w))]
pub unsafe fn __msa_vshf_w(a: v4i32, b: v4i32, c: v4i32) -> v4i32 {
    msa_vshf_w(a, mem::transmute(b), c)
}

/// Vector Data Preserving Shuffle
///
/// The vector shuffle instructions selectively copy data elements from the
/// concatenation of vectors `b` (two signed 64-bit integer numbers)
/// and `c` (two signed 64-bit integer numbers) in to vector `a`
/// (two signed 64-bit integer numbers) based on the corresponding control element in `a`.
/// The least significant 6 bits in `a` control elements modulo the number of elements in
/// the concatenated vectors `b`, `a` specify the index of the source element.
/// If bit 6 or bit 7 is 1, there will be no copy, but rather the destination element is set to 0.
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(vshf.d))]
pub unsafe fn __msa_vshf_d(a: v2i64, b: v2i64, c: v2i64) -> v2i64 {
    msa_vshf_d(a, mem::transmute(b), c)
}

/// Vector Logical Exclusive Or
///
/// Each bit of vector `a` (sixteen unsigned 8-bit integer numbers)
/// is combined with the corresponding bit of vector `b` (sixteen unsigned 8-bit integer numbers)
/// in a bitwise logical XOR operation. The result is written to vector
/// (sixteen unsigned 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(xor.v))]
pub unsafe fn __msa_xor_v(a: v16u8, b: v16u8) -> v16u8 {
    msa_xor_v(a, mem::transmute(b))
}

/// Immediate Logical Exclusive Or
///
/// Each byte of vector `a` (sixteen unsigned 8-bit integer numbers)
/// is combined with the 8-bit immediate `imm8`
/// in a bitwise logical XOR operation. The result is written to vector
/// (sixteen unsigned 8-bit integer numbers).
///
#[inline]
#[target_feature(enable = "msa")]
#[cfg_attr(test, assert_instr(xori.b, imm8 = 0b11111111))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn __msa_xori_b<const IMM8: i32>(a: v16u8) -> v16u8 {
    static_assert_imm8!(IMM8);
    msa_xori_b(a, IMM8)
}

#[cfg(test)]
mod tests {
    use crate::{
        core_arch::{mips::msa::*, simd::*},
        mem,
    };
    use std::{f32, f64};
    use stdarch_test::simd_test;

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_add_a_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            1, 2, 3, 4,
            1, 2, 3, 4,
            1, 2, 3, 4,
            1, 2, 3, 4
        );
        #[rustfmt::skip]
        let b = i8x16::new(
            -4, -3, -2, -1,
            -4, -3, -2, -1,
            -4, -3, -2, -1,
            -4, -3, -2, -1
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            5, 5, 5, 5,
            5, 5, 5, 5, 
            5, 5, 5, 5, 
            5, 5, 5, 5
        );

        assert_eq!(
            r,
            mem::transmute(__msa_add_a_b(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_add_a_h() {
        #[rustfmt::skip]
        let a = i16x8::new(1, 2, 3, 4, 1, 2, 3, 4);
        #[rustfmt::skip]
        let b = i16x8::new(-4, -3, -2, -1, -4, -3, -2, -1);
        #[rustfmt::skip]
        let r = i16x8::new(5, 5, 5, 5, 5, 5, 5, 5);

        assert_eq!(
            r,
            mem::transmute(__msa_add_a_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_add_a_w() {
        #[rustfmt::skip]
        let a = i32x4::new(1, 2, 3, 4);
        #[rustfmt::skip]
        let b = i32x4::new(-4, -3, -2, -1);
        #[rustfmt::skip]
        let r = i32x4::new(5, 5, 5, 5);

        assert_eq!(
            r,
            mem::transmute(__msa_add_a_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_add_a_d() {
        #[rustfmt::skip]
        let a = i64x2::new(1, 2);
        #[rustfmt::skip]
        let b = i64x2::new(-4, -3);
        #[rustfmt::skip]
        let r = i64x2::new(5, 5);

        assert_eq!(
            r,
            mem::transmute(__msa_add_a_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_adds_a_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            100, i8::MAX, 100, i8::MAX,
            100, i8::MAX, 100, i8::MAX,
            100, i8::MAX, 100, i8::MAX,
            100, i8::MAX, 100, i8::MAX
        );
        #[rustfmt::skip]
        let b = i8x16::new(
            -4, -3, -2, -100,
            -4, -3, -2, -100,
            -4, -3, -2, -100,
            -4, -3, -2, -100
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            104, 127, 102, 127, 
            104, 127, 102, 127, 
            104, 127, 102, 127, 
            104, 127, 102, 127
        );

        assert_eq!(
            r,
            mem::transmute(__msa_adds_a_b(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_adds_a_h() {
        #[rustfmt::skip]
        let a = i16x8::new(
            100, i16::MAX, 100, i16::MAX, 
            100, i16::MAX, 100, i16::MAX
        );
        #[rustfmt::skip]
        let b = i16x8::new(-4, -3, -2, -1, -4, -3, -2, -1);
        #[rustfmt::skip]
        let r = i16x8::new(
            104, i16::MAX, 102, i16::MAX,
            104, i16::MAX, 102, i16::MAX
        );

        assert_eq!(
            r,
            mem::transmute(__msa_adds_a_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_adds_a_w() {
        #[rustfmt::skip]
        let a = i32x4::new(100, i32::MAX, 100, i32::MAX);
        #[rustfmt::skip]
        let b = i32x4::new(-4, -3, -2, -1);
        #[rustfmt::skip]
        let r = i32x4::new(104, i32::MAX, 102, i32::MAX);

        assert_eq!(
            r,
            mem::transmute(__msa_adds_a_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_adds_a_d() {
        #[rustfmt::skip]
        let a = i64x2::new(100, i64::MAX);
        #[rustfmt::skip]
        let b = i64x2::new(-4, -3);
        #[rustfmt::skip]
        let r = i64x2::new(104, i64::MAX);

        assert_eq!(
            r,
            mem::transmute(__msa_adds_a_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_adds_s_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            100, i8::MIN, 100, i8::MAX,
            100, i8::MIN, 100, i8::MAX,
            100, i8::MIN, 100, i8::MAX,
            100, i8::MIN, 100, i8::MAX
        );
        #[rustfmt::skip]
        let b = i8x16::new(
            -4, -3, -2, 100,
            -4, -3, -2, 100,
            -4, -3, -2, 100,
            -4, -3, -2, 100
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            96, i8::MIN, 98, i8::MAX, 
            96, i8::MIN, 98, i8::MAX, 
            96, i8::MIN, 98, i8::MAX, 
            96, i8::MIN, 98, i8::MAX
        );

        assert_eq!(
            r,
            mem::transmute(__msa_adds_s_b(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_adds_s_h() {
        #[rustfmt::skip]
        let a = i16x8::new(
            100, i16::MIN, 100, i16::MAX, 
            100, i16::MIN, 100, i16::MAX
        );
        #[rustfmt::skip]
        let b = i16x8::new(-4, -3, -2, 1, -4, -3, -2, 1);
        #[rustfmt::skip]
        let r = i16x8::new(
            96, i16::MIN, 98, i16::MAX, 
            96, i16::MIN, 98, i16::MAX
        );

        assert_eq!(
            r,
            mem::transmute(__msa_adds_s_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_adds_s_w() {
        #[rustfmt::skip]
        let a = i32x4::new(100, i32::MAX, 100, i32::MIN);
        #[rustfmt::skip]
        let b = i32x4::new(-4, 3, -2, -1);
        #[rustfmt::skip]
        let r = i32x4::new(96, i32::MAX, 98, i32::MIN);

        assert_eq!(
            r,
            mem::transmute(__msa_adds_s_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_adds_s_d() {
        #[rustfmt::skip]
        let a = i64x2::new(100, i64::MIN);
        #[rustfmt::skip]
        let b = i64x2::new(-4, -3);
        #[rustfmt::skip]
        let r = i64x2::new(96, i64::MIN);

        assert_eq!(
            r,
            mem::transmute(__msa_adds_s_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_adds_u_b() {
        #[rustfmt::skip]
        let a = u8x16::new(
            100, u8::MAX, 100, u8::MAX,
            100, u8::MAX, 100, u8::MAX,
            100, u8::MAX, 100, u8::MAX,
            100, u8::MAX, 100, u8::MAX
        );
        #[rustfmt::skip]
        let b = u8x16::new(
            4, 3, 2, 100,
            4, 3, 2, 100,
            4, 3, 2, 100,
            4, 3, 2, 100
        );
        #[rustfmt::skip]
        let r = u8x16::new(
            104, u8::MAX, 102, u8::MAX, 
            104, u8::MAX, 102, u8::MAX, 
            104, u8::MAX, 102, u8::MAX, 
            104, u8::MAX, 102, u8::MAX
        );

        assert_eq!(
            r,
            mem::transmute(__msa_adds_u_b(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_adds_u_h() {
        #[rustfmt::skip]
        let a = u16x8::new(
            100, u16::MAX, 100, u16::MAX, 
            100, u16::MAX, 100, u16::MAX
        );
        #[rustfmt::skip]
        let b = u16x8::new(4, 3, 2, 1, 4, 3, 2, 1);
        #[rustfmt::skip]
        let r = u16x8::new(
            104, u16::MAX, 102, u16::MAX, 
            104, u16::MAX, 102, u16::MAX
        );

        assert_eq!(
            r,
            mem::transmute(__msa_adds_u_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_adds_u_w() {
        #[rustfmt::skip]
        let a = u32x4::new(100, u32::MAX, 100, u32::MAX);
        #[rustfmt::skip]
        let b = u32x4::new(4, 3, 2, 1);
        #[rustfmt::skip]
        let r = u32x4::new(104, u32::MAX, 102, u32::MAX);

        assert_eq!(
            r,
            mem::transmute(__msa_adds_u_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_adds_u_d() {
        #[rustfmt::skip]
        let a = u64x2::new(100, u64::MAX);
        #[rustfmt::skip]
        let b = u64x2::new(4, 3);
        #[rustfmt::skip]
        let r = u64x2::new(104, u64::MAX);

        assert_eq!(
            r,
            mem::transmute(__msa_adds_u_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_addv_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            100, i8::MIN, 100, i8::MAX,
            100, i8::MIN, 100, i8::MAX,
            100, i8::MIN, 100, i8::MAX,
            100, i8::MIN, 100, i8::MAX
        );
        #[rustfmt::skip]
        let b = i8x16::new(
            -4, -3, -2, 100,
            -4, -3, -2, 100,
            -4, -3, -2, 100,
            -4, -3, -2, 100
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            96, 125, 98, -29, 
            96, 125, 98, -29, 
            96, 125, 98, -29, 
            96, 125, 98, -29
        );

        assert_eq!(
            r,
            mem::transmute(__msa_addv_b(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_addv_h() {
        #[rustfmt::skip]
        let a = i16x8::new(
            100, i16::MIN, 100, i16::MAX, 
            100, i16::MIN, 100, i16::MAX
        );
        #[rustfmt::skip]
        let b = i16x8::new(-4, -3, -2, 1, -4, -3, -2, 1);
        #[rustfmt::skip]
        let r = i16x8::new(96, 32765, 98, -32768, 96, 32765, 98, -32768);

        assert_eq!(
            r,
            mem::transmute(__msa_addv_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_addv_w() {
        #[rustfmt::skip]
        let a = i32x4::new(100, i32::MAX, 100, i32::MIN);
        #[rustfmt::skip]
        let b = i32x4::new(-4, 3, -2, -1);
        #[rustfmt::skip]
        let r = i32x4::new(96, -2147483646, 98, 2147483647);

        assert_eq!(
            r,
            mem::transmute(__msa_addv_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_addv_d() {
        #[rustfmt::skip]
        let a = i64x2::new(100, i64::MIN);
        #[rustfmt::skip]
        let b = i64x2::new(-4, -3);
        #[rustfmt::skip]
        let r = i64x2::new(96, 9223372036854775805);

        assert_eq!(
            r,
            mem::transmute(__msa_addv_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_addvi_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            100, i8::MAX, 100, i8::MAX,
            100, i8::MAX, 100, i8::MAX,
            100, i8::MAX, 100, i8::MAX,
            100, i8::MAX, 100, i8::MAX
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            103, -126, 103, -126, 
            103, -126, 103, -126, 
            103, -126, 103, -126, 
            103, -126, 103, -126
        );

        assert_eq!(r, mem::transmute(__msa_addvi_b(mem::transmute(a), 67)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_addvi_h() {
        #[rustfmt::skip]
        let a = i16x8::new(
            i16::MAX, 3276, -100, -127, 
            i16::MAX, 3276, -100, -127
        );
        #[rustfmt::skip]
        let r = i16x8::new(
            -32766, 3279, -97, -124, 
            -32766, 3279, -97, -124
        );

        assert_eq!(r, mem::transmute(__msa_addvi_h(mem::transmute(a), 67)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_addvi_w() {
        #[rustfmt::skip]
        let a = i32x4::new(100, i32::MAX, 100, i32::MIN);
        #[rustfmt::skip]
        let r = i32x4::new(103, -2147483646, 103, -2147483645);

        assert_eq!(r, mem::transmute(__msa_addvi_w(mem::transmute(a), 67)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_addvi_d() {
        #[rustfmt::skip]
        let a = i64x2::new(100, i64::MIN);
        #[rustfmt::skip]
        let r = i64x2::new(117, -9223372036854775791);

        assert_eq!(r, mem::transmute(__msa_addvi_d(mem::transmute(a), 17)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_and_v() {
        #[rustfmt::skip]
        let a = u8x16::new(
            100, u8::MAX, 100, u8::MAX,
            100, u8::MAX, 100, u8::MAX,
            100, u8::MAX, 100, u8::MAX,
            100, u8::MAX, 100, u8::MAX
    );
        #[rustfmt::skip]
        let b = u8x16::new(
            4, 3, 2, 100,
            4, 3, 2, 100,
            4, 3, 2, 100,
            4, 3, 2, 100
        );
        #[rustfmt::skip]
        let r = u8x16::new(
            4, 3, 0, 100, 
            4, 3, 0, 100, 
            4, 3, 0, 100, 
            4, 3, 0, 100
        );

        assert_eq!(
            r,
            mem::transmute(__msa_and_v(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_andi_b() {
        #[rustfmt::skip]
        let a = u8x16::new(
            100, u8::MAX, 100, u8::MAX,
            100, u8::MAX, 100, u8::MAX,
            100, u8::MAX, 100, u8::MAX,
            100, u8::MAX, 100, u8::MAX
        );
        #[rustfmt::skip]
        let r = u8x16::new(
            4, 5, 4, 5, 
            4, 5, 4, 5, 
            4, 5, 4, 5, 
            4, 5, 4, 5
        );

        assert_eq!(r, mem::transmute(__msa_andi_b(mem::transmute(a), 5)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_asub_s_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            -1, -2, -3, -4,
            -1, -2, -3, -4,
            -1, -2, -3, -4,
            -1, -2, -3, -4
        );
        #[rustfmt::skip]
        let b = i8x16::new(
            -6, -7, -8, -9,
            -6, -7, -8, -9,
            -6, -7, -8, -9,
            -6, -7, -8, -9
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            5, 5, 5, 5, 
            5, 5, 5, 5, 
            5, 5, 5, 5, 
            5, 5, 5, 5
        );

        assert_eq!(
            r,
            mem::transmute(__msa_asub_s_b(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_asub_s_h() {
        #[rustfmt::skip]
        let a = i16x8::new(-1, -2, -3, -4, -1, -2, -3, -4);
        #[rustfmt::skip]
        let b = i16x8::new(-6, -7, -8, -9, -6, -7, -8, -9);
        #[rustfmt::skip]
        let r = i16x8::new(5, 5, 5, 5, 5, 5, 5, 5);

        assert_eq!(
            r,
            mem::transmute(__msa_asub_s_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_asub_s_w() {
        #[rustfmt::skip]
        let a = i32x4::new(-1, -2, -3, -4);
        #[rustfmt::skip]
        let b = i32x4::new(-6, -7, -8, -9);
        #[rustfmt::skip]
        let r = i32x4::new(5, 5, 5, 5);

        assert_eq!(
            r,
            mem::transmute(__msa_asub_s_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_asub_s_d() {
        #[rustfmt::skip]
        let a = i64x2::new(-1, -2);
        #[rustfmt::skip]
        let b = i64x2::new(-6, -7);
        #[rustfmt::skip]
        let r = i64x2::new(5, 5);

        assert_eq!(
            r,
            mem::transmute(__msa_asub_s_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_asub_u_b() {
        #[rustfmt::skip]
        let a = u8x16::new(
            1, 2, 3, 4,
            1, 2, 3, 4,
            1, 2, 3, 4,
            1, 2, 3, 4
        );
        #[rustfmt::skip]
        let b = u8x16::new(
            6, 7, 8, 9,
            6, 7, 8, 9,
            6, 7, 8, 9,
            6, 7, 8, 9
        );
        #[rustfmt::skip]
        let r = u8x16::new(
            5, 5, 5, 5, 
            5, 5, 5, 5, 
            5, 5, 5, 5, 
            5, 5, 5, 5
        );

        assert_eq!(
            r,
            mem::transmute(__msa_asub_u_b(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_asub_u_h() {
        #[rustfmt::skip]
        let a = u16x8::new(1, 2, 3, 4, 1, 2, 3, 4);
        #[rustfmt::skip]
        let b = u16x8::new(6, 7, 8, 9, 6, 7, 8, 9);
        #[rustfmt::skip]
        let r = u16x8::new(5, 5, 5, 5, 5, 5, 5, 5);

        assert_eq!(
            r,
            mem::transmute(__msa_asub_u_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_asub_u_w() {
        #[rustfmt::skip]
        let a = u32x4::new(1, 2, 3, 4);
        #[rustfmt::skip]
        let b = u32x4::new(6, 7, 8, 9);
        #[rustfmt::skip]
        let r = u32x4::new(5, 5, 5, 5);

        assert_eq!(
            r,
            mem::transmute(__msa_asub_u_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_asub_u_d() {
        #[rustfmt::skip]
        let a = u64x2::new(1, 2);
        #[rustfmt::skip]
        let b = u64x2::new(6, 7);
        #[rustfmt::skip]
        let r = u64x2::new(5, 5);

        assert_eq!(
            r,
            mem::transmute(__msa_asub_u_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ave_s_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            -1, -2, -3, -4,
            -1, -2, -3, -4,
            -1, -2, -3, -4,
            -1, -2, -3, -4
        );
        #[rustfmt::skip]
        let b = i8x16::new(
            6, -7, 8, -9,
            6, -7, 8, -9,
            6, -7, 8, -9,
            6, -7, 8, -9
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            2, -5, 2, -7, 
            2, -5, 2, -7, 
            2, -5, 2, -7, 
            2, -5, 2, -7
        );

        assert_eq!(
            r,
            mem::transmute(__msa_ave_s_b(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ave_s_h() {
        #[rustfmt::skip]
        let a = i16x8::new(-1, -2, -3, -4, -1, -2, -3, -4);
        #[rustfmt::skip]
        let b = i16x8::new(6, -7, 8, -9, 6, -7, 8, -9);
        #[rustfmt::skip]
        let r = i16x8::new(2, -5, 2, -7, 2, -5, 2, -7);

        assert_eq!(
            r,
            mem::transmute(__msa_ave_s_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ave_s_w() {
        #[rustfmt::skip]
        let a = i32x4::new(-1, -2, -3, -4);
        #[rustfmt::skip]
        let b = i32x4::new(6, -7, 8, -9);
        #[rustfmt::skip]
        let r = i32x4::new(2, -5, 2, -7);

        assert_eq!(
            r,
            mem::transmute(__msa_ave_s_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ave_s_d() {
        #[rustfmt::skip]
        let a = i64x2::new(-1, -2);
        #[rustfmt::skip]
        let b = i64x2::new(-6, -7);
        #[rustfmt::skip]
        let r = i64x2::new(-4, -5);

        assert_eq!(
            r,
            mem::transmute(__msa_ave_s_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ave_u_b() {
        #[rustfmt::skip]
        let a = u8x16::new(
            1, 2, 3, 4,
            1, 2, 3, 4,
            1, 2, 3, 4,
            1, 2, 3, 4
        );
        #[rustfmt::skip]
        let b = u8x16::new(
            6, 7, 8, 9,
            6, 7, 8, 9,
            6, 7, 8, 9,
            6, 7, 8, 9
        );
        #[rustfmt::skip]
        let r = u8x16::new(
            3, 4, 5, 6, 
            3, 4, 5, 6, 
            3, 4, 5, 6, 
            3, 4, 5, 6
        );

        assert_eq!(
            r,
            mem::transmute(__msa_ave_u_b(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ave_u_h() {
        #[rustfmt::skip]
        let a = u16x8::new(1, 2, 3, 4, 1, 2, 3, 4);
        #[rustfmt::skip]
        let b = u16x8::new(6, 7, 8, 9, 6, 7, 8, 9);
        #[rustfmt::skip]
        let r = u16x8::new(3, 4, 5, 6, 3, 4, 5, 6);

        assert_eq!(
            r,
            mem::transmute(__msa_ave_u_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ave_u_w() {
        #[rustfmt::skip]
        let a = u32x4::new(1, 2, 3, 4);
        #[rustfmt::skip]
        let b = u32x4::new(6, 7, 8, 9);
        #[rustfmt::skip]
        let r = u32x4::new(3, 4, 5, 6);

        assert_eq!(
            r,
            mem::transmute(__msa_ave_u_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ave_u_d() {
        #[rustfmt::skip]
        let a = u64x2::new(1, 2);
        #[rustfmt::skip]
        let b = u64x2::new(6, 7);
        #[rustfmt::skip]
        let r = u64x2::new(3, 4);

        assert_eq!(
            r,
            mem::transmute(__msa_ave_u_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_aver_s_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            -1, -2, 3, -4,
            -1, -2, 3, -4,
            -1, -2, 3, -4,
            -1, -2, 3, -4
        );
        #[rustfmt::skip]
        let b = i8x16::new(
            -6, 7, -8, -9,
            -6, 7, -8, -9,
            -6, 7, -8, -9,
            -6, 7, -8, -9
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            -3, 3, -2, -6, 
            -3, 3, -2, -6, 
            -3, 3, -2, -6, 
            -3, 3, -2, -6
        );

        assert_eq!(
            r,
            mem::transmute(__msa_aver_s_b(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_aver_s_h() {
        #[rustfmt::skip]
        let a = i16x8::new(-1, -2, 3, -4, -1, -2, 3, -4);
        #[rustfmt::skip]
        let b = i16x8::new(-6, 7, -8, -9, -6, 7, -8, -9);
        #[rustfmt::skip]
        let r = i16x8::new(-3, 3, -2, -6, -3, 3, -2, -6);

        assert_eq!(
            r,
            mem::transmute(__msa_aver_s_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_aver_s_w() {
        #[rustfmt::skip]
        let a = i32x4::new(-1, -2, 3, -4);
        #[rustfmt::skip]
        let b = i32x4::new(-6, 7, -8, -9);
        #[rustfmt::skip]
        let r = i32x4::new(-3, 3, -2, -6);

        assert_eq!(
            r,
            mem::transmute(__msa_aver_s_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_aver_s_d() {
        #[rustfmt::skip]
        let a = i64x2::new(-1, -2);
        #[rustfmt::skip]
        let b = i64x2::new(-6, -7);
        #[rustfmt::skip]
        let r = i64x2::new(-3, -4);

        assert_eq!(
            r,
            mem::transmute(__msa_aver_s_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_aver_u_b() {
        #[rustfmt::skip]
        let a = u8x16::new(
            1, 2, 3, 4,
            1, 2, 3, 4,
            1, 2, 3, 4,
            1, 2, 3, 4
        );
        #[rustfmt::skip]
        let b = u8x16::new(
            6, 7, 8, 9,
            6, 7, 8, 9,
            6, 7, 8, 9,
            6, 7, 8, 9
        );
        #[rustfmt::skip]
        let r = u8x16::new(
            4, 5, 6, 7, 
            4, 5, 6, 7, 
            4, 5, 6, 7, 
            4, 5, 6, 7
        );

        assert_eq!(
            r,
            mem::transmute(__msa_aver_u_b(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_aver_u_h() {
        #[rustfmt::skip]
        let a = u16x8::new(1, 2, 3, 4, 1, 2, 3, 4);
        #[rustfmt::skip]
        let b = u16x8::new(6, 7, 8, 9, 6, 7, 8, 9);
        #[rustfmt::skip]
        let r = u16x8::new(4, 5, 6, 7, 4, 5, 6, 7);

        assert_eq!(
            r,
            mem::transmute(__msa_aver_u_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_aver_u_w() {
        #[rustfmt::skip]
        let a = u32x4::new(1, 2, 3, 4);
        #[rustfmt::skip]
        let b = u32x4::new(6, 7, 8, 9);
        #[rustfmt::skip]
        let r = u32x4::new(4, 5, 6, 7);

        assert_eq!(
            r,
            mem::transmute(__msa_aver_u_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_aver_u_d() {
        #[rustfmt::skip]
        let a = u64x2::new(1, 2);
        #[rustfmt::skip]
        let b = u64x2::new(6, 7);
        #[rustfmt::skip]
        let r = u64x2::new(4, 5);

        assert_eq!(
            r,
            mem::transmute(__msa_aver_u_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_bclr_b() {
        #[rustfmt::skip]
        let a = u8x16::new(
            255, 155, 55, 1,
            255, 155, 55, 1,
            255, 155, 55, 1,
            255, 155, 55, 1
        );
        #[rustfmt::skip]
        let b = u8x16::new(
            6, 7, 8, 9,
            6, 7, 8, 9,
            6, 7, 8, 9,
            6, 7, 8, 9
        );
        #[rustfmt::skip]
        let r = u8x16::new(
            191, 27, 54, 1, 
            191, 27, 54, 1, 
            191, 27, 54, 1, 
            191, 27, 54, 1
        );

        assert_eq!(
            r,
            mem::transmute(__msa_bclr_b(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_bclr_h() {
        #[rustfmt::skip]
        let a = u16x8::new(255, 155, 55, 1, 255, 155, 55, 1);
        #[rustfmt::skip]
        let b = u16x8::new(6, 7, 8, 9, 6, 7, 8, 9);
        #[rustfmt::skip]
        let r = u16x8::new(191, 27, 55, 1, 191, 27, 55, 1);

        assert_eq!(
            r,
            mem::transmute(__msa_bclr_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_bclr_w() {
        #[rustfmt::skip]
        let a = u32x4::new(255, 155, 55, 1);
        #[rustfmt::skip]
        let b = u32x4::new(6, 7, 8, 9);
        #[rustfmt::skip]
        let r = u32x4::new(191, 27, 55, 1);

        assert_eq!(
            r,
            mem::transmute(__msa_bclr_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_bclr_d() {
        #[rustfmt::skip]
        let a = u64x2::new(255, 155);
        #[rustfmt::skip]
        let b = u64x2::new(6, 7);
        #[rustfmt::skip]
        let r = u64x2::new(191, 27);

        assert_eq!(
            r,
            mem::transmute(__msa_bclr_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_bclri_b() {
        #[rustfmt::skip]
        let a = u8x16::new(
            255, 155, 55, 1,
            255, 155, 55, 1,
            255, 155, 55, 1,
            255, 155, 55, 1
        );
        #[rustfmt::skip]
        let r = u8x16::new(
            247, 147, 55, 1, 
            247, 147, 55, 1, 
            247, 147, 55, 1, 
            247, 147, 55, 1
        );

        assert_eq!(r, mem::transmute(__msa_bclri_b(mem::transmute(a), 3)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_bclri_h() {
        #[rustfmt::skip]
        let a = u16x8::new(2155, 1155, 155, 1, 2155, 1155, 155, 1);
        #[rustfmt::skip]
        let r = u16x8::new(107, 1155, 155, 1, 107, 1155, 155, 1);

        assert_eq!(r, mem::transmute(__msa_bclri_h(mem::transmute(a), 11)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_bclri_w() {
        #[rustfmt::skip]
        let a = u32x4::new(211111155, 111111155, 11111155, 1);
        #[rustfmt::skip]
        let r = u32x4::new(202722547, 102722547, 2722547, 1);

        assert_eq!(r, mem::transmute(__msa_bclri_w(mem::transmute(a), 23)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_bclri_d() {
        #[rustfmt::skip]
        let a = u64x2::new(211111111155, 11111111111111155);
        #[rustfmt::skip]
        let r = u64x2::new(73672157683, 11110973672157683);

        assert_eq!(r, mem::transmute(__msa_bclri_d(mem::transmute(a), 37)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_binsl_b() {
        #[rustfmt::skip]
        let a = u8x16::new(
            255, 155, 55, 1,
            255, 155, 55, 1,
            255, 155, 55, 1,
            255, 155, 55, 1
        );
        #[rustfmt::skip]
        let b = u8x16::new(
            6, 7, 8, 9,
            6, 7, 8, 9,
            6, 7, 8, 9,
            6, 7, 8, 9
        );
        #[rustfmt::skip]
        let c = u8x16::new(
            1, 3, 5, 9,
            1, 3, 5, 9,
            1, 3, 5, 9,
            1, 3, 5, 9
        );
        #[rustfmt::skip]
        let r = u8x16::new(
            63, 11, 11, 1, 
            63, 11, 11, 1, 
            63, 11, 11, 1, 
            63, 11, 11, 1
        );

        assert_eq!(
            r,
            mem::transmute(__msa_binsl_b(
                mem::transmute(a),
                mem::transmute(b),
                mem::transmute(c)
            ))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_binsl_h() {
        #[rustfmt::skip]
        let a = u16x8::new(
            32767, 16384, 8192, 4096, 
            32767, 16384, 8192, 4096
        );
        #[rustfmt::skip]
        let b = u16x8::new(
            21656, 5273, 7081, 2985, 
            21656, 5273, 7081, 2985
        );
        #[rustfmt::skip]
        let c = u16x8::new(
            3, 7, 9, 13, 
            15, 17, 21, 23
        );
        #[rustfmt::skip]
        let r = u16x8::new(
            24575, 5120, 7040, 2984, 
            21656, 0, 6144, 2816
        );

        assert_eq!(
            r,
            mem::transmute(__msa_binsl_h(
                mem::transmute(a),
                mem::transmute(b),
                mem::transmute(c)
            ))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_binsl_w() {
        #[rustfmt::skip]
        let a = u32x4::new(2147483647, 536870912, 67108864, 8388608);
        #[rustfmt::skip]
        let b = u32x4::new(1036372536, 259093134, 78219975, 1119499719);
        #[rustfmt::skip]
        let c = u32x4::new(11, 15, 31, 37);
        #[rustfmt::skip]
        let r = u32x4::new(1037041663, 259063808, 78219975, 1082130432);

        assert_eq!(
            r,
            mem::transmute(__msa_binsl_w(
                mem::transmute(a),
                mem::transmute(b),
                mem::transmute(c)
            ))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_binsl_d() {
        #[rustfmt::skip]
        let a = u64x2::new(8006399338, 2882303762);
        #[rustfmt::skip]
        let b = u64x2::new(9223372036854775805, 536870912);
        #[rustfmt::skip]
        let c = u64x2::new(12, 48);
        #[rustfmt::skip]
        let r = u64x2::new(9221120245047489898, 536901394);

        assert_eq!(
            r,
            mem::transmute(__msa_binsl_d(
                mem::transmute(a),
                mem::transmute(b),
                mem::transmute(c)
            ))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_binsli_b() {
        #[rustfmt::skip]
        let a = u8x16::new(
            u8::MAX, 155, 55, 1,
            u8::MAX, 155, 55, 1,
            u8::MAX, 155, 55, 1,
            u8::MAX, 155, 55, 1
        );
        #[rustfmt::skip]
        let b = u8x16::new(
            6, 7, 8, 9,
            6, 7, 8, 9,
            6, 7, 8, 9,
            6, 7, 8, 9
        );
        #[rustfmt::skip]
        let r = u8x16::new(
            7, 7, 11, 9, 
            7, 7, 11, 9, 
            7, 7, 11, 9, 
            7, 7, 11, 9
        );

        assert_eq!(
            r,
            mem::transmute(__msa_binsli_b(mem::transmute(a), mem::transmute(b), 5))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_binsli_h() {
        #[rustfmt::skip]
        let a = u16x8::new(
            32767, 16384, 8192, 4096, 
            32767, 16384, 8192, 4096
        );
        #[rustfmt::skip]
        let b = u16x8::new(
            21656, 5273, 7081, 2985, 
            21656, 5273, 7081, 2985
        );
        #[rustfmt::skip]
        let r = u16x8::new(
            21659, 5272, 7080, 2984, 
            21659, 5272, 7080, 2984
        );

        assert_eq!(
            r,
            mem::transmute(__msa_binsli_h(mem::transmute(a), mem::transmute(b), 13))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_binsli_w() {
        #[rustfmt::skip]
        let a = u32x4::new(2147483647, 536870912, 67108864, 8388608);
        #[rustfmt::skip]
        let b = u32x4::new(1036372536, 259093134, 78219975, 1119499719);
        #[rustfmt::skip]
        let r = u32x4::new(1036386303, 259080192, 78217216, 1119485952);

        assert_eq!(
            r,
            mem::transmute(__msa_binsli_w(mem::transmute(a), mem::transmute(b), 17))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_binsli_d() {
        #[rustfmt::skip]
        let a = u64x2::new(8006399338, 2882303762);
        #[rustfmt::skip]
        let b = u64x2::new(9223372036854775805, 536870912);
        #[rustfmt::skip]
        let r = u64x2::new(9223372036854773098, 536901394);

        assert_eq!(
            r,
            mem::transmute(__msa_binsli_d(mem::transmute(a), mem::transmute(b), 48))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_binsr_b() {
        #[rustfmt::skip]
        let a = u8x16::new(
            255, 155, 55, 1,
            255, 155, 55, 1,
            255, 155, 55, 1,
            255, 155, 55, 1
        );
        #[rustfmt::skip]
        let b = u8x16::new(
            6, 7, 8, 9,
            6, 7, 8, 9,
            6, 7, 8, 9,
            6, 7, 8, 9
        );
        #[rustfmt::skip]
        let c = u8x16::new(
            1, 3, 5, 9,
            1, 3, 5, 9,
            1, 3, 5, 9,
            1, 3, 5, 9
        );
        #[rustfmt::skip]
        let r = u8x16::new(
            254, 151, 8, 1, 
            254, 151, 8, 1, 
            254, 151, 8, 1, 
            254, 151, 8, 1
        );

        assert_eq!(
            r,
            mem::transmute(__msa_binsr_b(
                mem::transmute(a),
                mem::transmute(b),
                mem::transmute(c)
            ))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_binsr_h() {
        #[rustfmt::skip]
        let a = u16x8::new(
            32767, 16384, 8192, 4096, 
            32767, 16384, 8192, 4096
        );
        #[rustfmt::skip]
        let b = u16x8::new(
            21656, 5273, 7081, 2985, 
            21656, 5273, 7081, 2985
        );
        #[rustfmt::skip]
        let c = u16x8::new(
            3, 7, 9, 13, 
            15, 17, 21, 23
        );
        #[rustfmt::skip]
        let r = u16x8::new(
            32760, 16537, 9129, 2985, 
            21656, 16385, 8233, 4265
        );

        assert_eq!(
            r,
            mem::transmute(__msa_binsr_h(
                mem::transmute(a),
                mem::transmute(b),
                mem::transmute(c)
            ))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_binsr_w() {
        #[rustfmt::skip]
        let a = u32x4::new(2147483647, 536870912, 67108864, 8388608);
        #[rustfmt::skip]
        let b = u32x4::new(1036372536, 259093134, 78219975, 1119499719);
        #[rustfmt::skip]
        let c = u32x4::new(11, 15, 31, 37);
        #[rustfmt::skip]
        let r = u32x4::new(2147482168, 536900238, 78219975, 8388615);

        assert_eq!(
            r,
            mem::transmute(__msa_binsr_w(
                mem::transmute(a),
                mem::transmute(b),
                mem::transmute(c)
            ))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_binsr_d() {
        #[rustfmt::skip]
        let a = u64x2::new(8006399338, 2882303762);
        #[rustfmt::skip]
        let b = u64x2::new(9223372036854775805, 536870912);
        #[rustfmt::skip]
        let c = u64x2::new(12, 48);
        #[rustfmt::skip]
        let r = u64x2::new(8006402045, 536870912);

        assert_eq!(
            r,
            mem::transmute(__msa_binsr_d(
                mem::transmute(a),
                mem::transmute(b),
                mem::transmute(c)
            ))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_binsri_b() {
        #[rustfmt::skip]
        let a = u8x16::new(
            255, 155, 55, 1,
            255, 155, 55, 1,
            255, 155, 55, 1,
            255, 155, 55, 1
        );
        #[rustfmt::skip]
        let b = u8x16::new(
            6, 7, 8, 9,
            6, 7, 8, 9,
            6, 7, 8, 9,
            6, 7, 8, 9
        );
        #[rustfmt::skip]
        let r = u8x16::new(
            198, 135, 8, 9, 
            198, 135, 8, 9, 
            198, 135, 8, 9, 
            198, 135, 8, 9
        );

        assert_eq!(
            r,
            mem::transmute(__msa_binsri_b(mem::transmute(a), mem::transmute(b), 5))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_binsri_h() {
        #[rustfmt::skip]
        let a = u16x8::new(
            32767, 16384, 8192, 4096, 
            32767, 16384, 8192, 4096
        );
        #[rustfmt::skip]
        let b = u16x8::new(
            21656, 5273, 7081, 2985, 
            21656, 5273, 7081, 2985
        );
        #[rustfmt::skip]
        let r = u16x8::new(
            21656, 21657, 7081, 2985, 
            21656, 21657, 7081, 2985
        );

        assert_eq!(
            r,
            mem::transmute(__msa_binsri_h(mem::transmute(a), mem::transmute(b), 13))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_binsri_w() {
        #[rustfmt::skip]
        let a = u32x4::new(2147483647, 536870912, 67108864, 8388608);
        #[rustfmt::skip]
        let b = u32x4::new(1036372536, 259093134, 78219975, 1119499719);
        #[rustfmt::skip]
        let r = u32x4::new(2147338808, 536965774, 67209927, 8533447);

        assert_eq!(
            r,
            mem::transmute(__msa_binsri_w(mem::transmute(a), mem::transmute(b), 17))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_binsri_d() {
        #[rustfmt::skip]
        let a = u64x2::new(8006399338, 2882303762);
        #[rustfmt::skip]
        let b = u64x2::new(9223372036854775805, 536870912);
        #[rustfmt::skip]
        let r = u64x2::new(562949953421309, 536870912);

        assert_eq!(
            r,
            mem::transmute(__msa_binsri_d(mem::transmute(a), mem::transmute(b), 48))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_bmnz_v() {
        #[rustfmt::skip]
        let a = u8x16::new(
            u8::MAX, 155, 55, 1,
            u8::MAX, 155, 55, 1,
            u8::MAX, 155, 55, 1,
            u8::MAX, 155, 55, 1
        );
        #[rustfmt::skip]
        let b = u8x16::new(
            6, 7, 8, 9,
            6, 7, 8, 9,
            6, 7, 8, 9,
            6, 7, 8, 9,
        );
        #[rustfmt::skip]
        let c = u8x16::new(
            3, 5, 7, 1,
            3, 5, 7, 1,
            3, 5, 7, 1,
            3, 5, 7, 1
        );
        #[rustfmt::skip]
        let r = u8x16::new(
            254, 159, 48, 1, 
            254, 159, 48, 1, 
            254, 159, 48, 1, 
            254, 159, 48, 1
        );

        assert_eq!(
            r,
            mem::transmute(__msa_bmnz_v(
                mem::transmute(a),
                mem::transmute(b),
                mem::transmute(c)
            ))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_bmnzi_b() {
        #[rustfmt::skip]
        let a = u8x16::new(
            u8::MAX, 155, 55, 1,
            u8::MAX, 155, 55, 1,
            u8::MAX, 155, 55, 1,
            u8::MAX, 155, 55, 1
        );
        #[rustfmt::skip]
        let b = u8x16::new(
            1, u8::MAX, 155, 55,
            1, u8::MAX, 155, 55,
            1, u8::MAX, 155, 55,
            1, u8::MAX, 155, 55
        );
        #[rustfmt::skip]
        let r = u8x16::new(
            249, 159, 51, 7, 
            249, 159, 51, 7, 
            249, 159, 51, 7, 
            249, 159, 51, 7
        );

        assert_eq!(
            r,
            mem::transmute(__msa_bmnzi_b(mem::transmute(a), mem::transmute(b), 7))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_bmz_v() {
        #[rustfmt::skip]
        let a = u8x16::new(
            u8::MAX, 155, 55, 1,
            u8::MAX, 155, 55, 1,
            u8::MAX, 155, 55, 1,
            u8::MAX, 155, 55, 1
        );
        #[rustfmt::skip]
        let b = u8x16::new(
            6, 7, 8, 9,
            6, 7, 8, 9,
            6, 7, 8, 9,
            6, 7, 8, 9
        );
        #[rustfmt::skip]
        let c = u8x16::new(
            3, 5, 7, 1,
            3, 5, 7, 1,
            3, 5, 7, 1,
            3, 5, 7, 1
        );
        #[rustfmt::skip]
        let r = u8x16::new(
            7, 3, 15, 9, 
            7, 3, 15, 9, 
            7, 3, 15, 9, 
            7, 3, 15, 9
        );

        assert_eq!(
            r,
            mem::transmute(__msa_bmz_v(
                mem::transmute(a),
                mem::transmute(b),
                mem::transmute(c)
            ))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_bmzi_b() {
        #[rustfmt::skip]
        let a = u8x16::new(
            u8::MAX, 155, 55, 1,
            u8::MAX, 155, 55, 1,
            u8::MAX, 155, 55, 1,
            u8::MAX, 155, 55, 1
        );
        #[rustfmt::skip]
        let b = u8x16::new(
            1, 255, 155, 55,
            1, 255, 155, 55,
            1, 255, 155, 55,
            1, 255, 155, 55
        );
        #[rustfmt::skip]
        let r = u8x16::new(
            7, 251, 159, 49, 
            7, 251, 159, 49, 
            7, 251, 159, 49, 
            7, 251, 159, 49
        );

        assert_eq!(
            r,
            mem::transmute(__msa_bmzi_b(mem::transmute(a), mem::transmute(b), 7))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_bneg_b() {
        #[rustfmt::skip]
        let a = u8x16::new(
            255, 155, 55, 1,
            255, 155, 55, 1,
            255, 155, 55, 1,
            255, 155, 55, 1
        );
        #[rustfmt::skip]
        let b = u8x16::new(
            6, 7, 8, 9,
            6, 7, 8, 9,
            6, 7, 8, 9,
            6, 7, 8, 9
        );
        #[rustfmt::skip]
        let r = u8x16::new(
            191, 27, 54, 3, 
            191, 27, 54, 3, 
            191, 27, 54, 3, 
            191, 27, 54, 3
        );

        assert_eq!(
            r,
            mem::transmute(__msa_bneg_b(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_bneg_h() {
        #[rustfmt::skip]
        let a = u16x8::new(255, 155, 55, 1, 255, 155, 55, 1);
        #[rustfmt::skip]
        let b = u16x8::new(6, 7, 8, 9, 6, 7, 8, 9);
        #[rustfmt::skip]
        let r = u16x8::new(191, 27, 311, 513, 191, 27, 311, 513);

        assert_eq!(
            r,
            mem::transmute(__msa_bneg_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_bneg_w() {
        #[rustfmt::skip]
        let a = u32x4::new(255, 155, 55, 1);
        #[rustfmt::skip]
        let b = u32x4::new(6, 7, 8, 9);
        #[rustfmt::skip]
        let r = u32x4::new(191, 27, 311, 513);

        assert_eq!(
            r,
            mem::transmute(__msa_bneg_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_bneg_d() {
        #[rustfmt::skip]
        let a = u64x2::new(255, 155);
        #[rustfmt::skip]
        let b = u64x2::new(6, 7);
        #[rustfmt::skip]
        let r = u64x2::new(191, 27);

        assert_eq!(
            r,
            mem::transmute(__msa_bneg_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_bnegi_b() {
        #[rustfmt::skip]
        let a = u8x16::new(
            50, 100, 127, u8::MAX,
            50, 100, 127, u8::MAX,
            50, 100, 127, u8::MAX,
            50, 100, 127, u8::MAX
        );
        #[rustfmt::skip]
        let r = u8x16::new(
            34, 116, 111, 239, 
            34, 116, 111, 239, 
            34, 116, 111, 239, 
            34, 116, 111, 239
        );

        assert_eq!(r, mem::transmute(__msa_bnegi_b(mem::transmute(a), 4)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_bnegi_h() {
        #[rustfmt::skip]
        let a = u16x8::new(
            32767, 3276, 100, 127,
            32767, 3276, 100, 127
        );
        #[rustfmt::skip]
        let r = u16x8::new(
            30719, 1228, 2148, 2175, 
            30719, 1228, 2148, 2175
        );

        assert_eq!(r, mem::transmute(__msa_bnegi_h(mem::transmute(a), 11)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_bnegi_w() {
        #[rustfmt::skip]
        let a = u32x4::new(100, 2147483647, 100, 2147483648);
        #[rustfmt::skip]
        let r = u32x4::new(16777316, 2130706431, 16777316, 2164260864);

        assert_eq!(r, mem::transmute(__msa_bnegi_w(mem::transmute(a), 24)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_bnegi_d() {
        #[rustfmt::skip]
        let a = u64x2::new(100, 9223372036854775808);
        #[rustfmt::skip]
        let r = u64x2::new(4398046511204, 9223376434901286912);

        assert_eq!(r, mem::transmute(__msa_bnegi_d(mem::transmute(a), 42)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_bnz_b() {
        #[rustfmt::skip]
        let a = u8x16::new(
            1, 1, 1, 1,
            1, 1, 1, 1,
            2, 2, 2, 2,
            4, 4, 0, 4,
        );
        let r = 0 as i32;

        assert_eq!(r, mem::transmute(__msa_bnz_b(mem::transmute(a))));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_bnz_h() {
        #[rustfmt::skip]
        let a = u16x8::new(
            32767, 3276, 100, 127,
            32767, 0, 100, 127
        );
        let r = 0 as i32;

        assert_eq!(r, mem::transmute(__msa_bnz_h(mem::transmute(a))));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_bnz_w() {
        #[rustfmt::skip]
        let a = u32x4::new(100, 2147483647, 0, 2147483648);
        let r = 0 as i32;

        assert_eq!(r, mem::transmute(__msa_bnz_w(mem::transmute(a))));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_bnz_d() {
        #[rustfmt::skip]
        let a = u64x2::new(100, 9223372036854775808);
        #[rustfmt::skip]
        let r = 1 as i32;

        assert_eq!(r, mem::transmute(__msa_bnz_d(mem::transmute(a))));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_bnz_v() {
        #[rustfmt::skip]
        let a = u8x16::new(
            0, 0, 0, 1,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
        );
        let r = 1 as i32;

        assert_eq!(r, mem::transmute(__msa_bnz_v(mem::transmute(a))));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_bsel_v() {
        #[rustfmt::skip]
        let a = u8x16::new(
            3, 5, 7, 1,
            3, 5, 7, 1,
            3, 5, 7, 1,
            3, 5, 7, 1
        );
        #[rustfmt::skip]
        let b = u8x16::new(
            6, 7, 8, 9,
            6, 7, 8, 9,
            6, 7, 8, 9,
            6, 7, 8, 9
        );
        #[rustfmt::skip]
        let c = u8x16::new(
            255, 155, 55, 1,
            255, 155, 55, 1,
            255, 155, 55, 1,
            255, 155, 55, 1
        );
        #[rustfmt::skip]
        let r = u8x16::new(
            7, 3, 15, 9, 
            7, 3, 15, 9, 
            7, 3, 15, 9, 
            7, 3, 15, 9
        );

        assert_eq!(
            r,
            mem::transmute(__msa_bsel_v(
                mem::transmute(a),
                mem::transmute(b),
                mem::transmute(c)
            ))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_bseli_b() {
        #[rustfmt::skip]
        let a = u8x16::new(
            255, 155, 55, 1,
            255, 155, 55, 1,
            255, 155, 55, 1,
            255, 155, 55, 1
        );
        #[rustfmt::skip]
        let b = u8x16::new(
            6, 7, 8, 9,
            6, 7, 8, 9,
            6, 7, 8, 9,
            6, 7, 8, 9
        );
        #[rustfmt::skip]
        let r = u8x16::new(
            121, 29, 57, 9, 
            121, 29, 57, 9, 
            121, 29, 57, 9, 
            121, 29, 57, 9
        );

        assert_eq!(
            r,
            mem::transmute(__msa_bseli_b(mem::transmute(a), mem::transmute(b), 121))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_bset_b() {
        #[rustfmt::skip]
        let a = u8x16::new(
            255, 155, 55, 1,
            255, 155, 55, 1,
            255, 155, 55, 1,
            255, 155, 55, 1
        );
        #[rustfmt::skip]
        let b = u8x16::new(
            6, 7, 8, 9,
            6, 7, 8, 9,
            6, 7, 8, 9,
            6, 7, 8, 9
        );
        #[rustfmt::skip]
        let r = u8x16::new(
            255, 155, 55, 3, 
            255, 155, 55, 3, 
            255, 155, 55, 3, 
            255, 155, 55, 3
        );

        assert_eq!(
            r,
            mem::transmute(__msa_bset_b(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_bset_h() {
        #[rustfmt::skip]
        let a = u16x8::new(255, 155, 55, 1, 255, 155, 55, 1);
        #[rustfmt::skip]
        let b = u16x8::new(6, 7, 8, 9, 6, 7, 8, 9);
        #[rustfmt::skip]
        let r = u16x8::new(255, 155, 311, 513, 255, 155, 311, 513);

        assert_eq!(
            r,
            mem::transmute(__msa_bset_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_bset_w() {
        #[rustfmt::skip]
        let a = u32x4::new(255, 155, 55, 1);
        #[rustfmt::skip]
        let b = u32x4::new(6, 7, 8, 9);
        #[rustfmt::skip]
        let r = u32x4::new(255, 155, 311, 513);

        assert_eq!(
            r,
            mem::transmute(__msa_bset_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_bset_d() {
        #[rustfmt::skip]
        let a = u64x2::new(255, 155);
        #[rustfmt::skip]
        let b = u64x2::new(6, 7);
        #[rustfmt::skip]
        let r = u64x2::new(255, 155);

        assert_eq!(
            r,
            mem::transmute(__msa_bset_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_bseti_b() {
        #[rustfmt::skip]
        let a = u8x16::new(
            255, 155, 55, 1,
            255, 155, 55, 1,
            255, 155, 55, 1,
            255, 155, 55, 1
        );
        #[rustfmt::skip]
        let r = u8x16::new(
            255, 159, 55, 5, 
            255, 159, 55, 5, 
            255, 159, 55, 5, 
            255, 159, 55, 5
        );

        assert_eq!(r, mem::transmute(__msa_bseti_b(mem::transmute(a), 2)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_bseti_h() {
        #[rustfmt::skip]
        let a = u16x8::new(255, 155, 55, 1, 255, 155, 55, 1);
        #[rustfmt::skip]
        let r = u16x8::new(255, 159, 55, 5, 255, 159, 55, 5);

        assert_eq!(r, mem::transmute(__msa_bseti_h(mem::transmute(a), 2)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_bseti_w() {
        #[rustfmt::skip]
        let a = u32x4::new(255, 155, 55, 1);
        #[rustfmt::skip]
        let r = u32x4::new(255, 159, 55, 5);

        assert_eq!(r, mem::transmute(__msa_bseti_w(mem::transmute(a), 2)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_bseti_d() {
        #[rustfmt::skip]
        let a = u64x2::new(255, 155);
        #[rustfmt::skip]
        let r = u64x2::new(255, 159);

        assert_eq!(r, mem::transmute(__msa_bseti_d(mem::transmute(a), 2)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_bz_b() {
        #[rustfmt::skip]
        let a = u8x16::new(
            255, 155, 55, 1,
            255, 155, 55, 1,
            255, 155, 55, 1,
            255, 155, 55, 1
        );
        let r = 0 as i32;

        assert_eq!(r, mem::transmute(__msa_bz_b(mem::transmute(a))));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_bz_h() {
        #[rustfmt::skip]
        let a = u16x8::new(0, 0, 0, 0, 0, 0, 0, 0);
        let r = 1 as i32;

        assert_eq!(r, mem::transmute(__msa_bz_h(mem::transmute(a))));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_bz_w() {
        #[rustfmt::skip]
        let a = u32x4::new(255, 0, 55, 1);
        let r = 1 as i32;

        assert_eq!(r, mem::transmute(__msa_bz_w(mem::transmute(a))));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_bz_d() {
        #[rustfmt::skip]
        let a = u64x2::new(255, 0);
        let r = 1 as i32;

        assert_eq!(r, mem::transmute(__msa_bz_d(mem::transmute(a))));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_bz_v() {
        #[rustfmt::skip]
        let a = u8x16::new(
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0
        );
        let r = 1 as i32;

        assert_eq!(r, mem::transmute(__msa_bz_v(mem::transmute(a))));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ceq_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            -128, 127, 55, 1,
            -128, 127, 55, 1,
            -128, 127, 55, 1,
            -128, 127, 55, 1
        );
        #[rustfmt::skip]
        let b = i8x16::new(
            -128, 126, 55, 1,
            -128, 126, 55, 1,
            -128, 126, 55, 1,
            -128, 126, 55, 1
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            -1, 0, -1, -1, 
            -1, 0, -1, -1, 
            -1, 0, -1, -1, 
            -1, 0, -1, -1
        );

        assert_eq!(
            r,
            mem::transmute(__msa_ceq_b(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ceq_h() {
        #[rustfmt::skip]
        let a = i16x8::new(255, 155, 55, 1, 255, 155, 55, 1);
        #[rustfmt::skip]
        let b = i16x8::new(255, 155, 56, 1, 255, 155, 56, 1);
        #[rustfmt::skip]
        let r = i16x8::new(-1, -1, 0, -1, -1, -1, 0, -1);

        assert_eq!(
            r,
            mem::transmute(__msa_ceq_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ceq_w() {
        #[rustfmt::skip]
        let a = i32x4::new(255, 155, 55, 1);
        #[rustfmt::skip]
        let b = i32x4::new(255, 156, 55, 1);
        #[rustfmt::skip]
        let r = i32x4::new(-1, 0, -1, -1);

        assert_eq!(
            r,
            mem::transmute(__msa_ceq_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ceq_d() {
        #[rustfmt::skip]
        let a = i64x2::new(255, 155);
        #[rustfmt::skip]
        let b = i64x2::new(255, 156);
        #[rustfmt::skip]
        let r = i64x2::new(-1, 0);

        assert_eq!(
            r,
            mem::transmute(__msa_ceq_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ceqi_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            100, -1, -4, 15,
            100, -1, -4, 15,
            100, -1, -4, 15,
            100, -1, -4, 15
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            0, 0, -1, 0,
            0, 0, -1, 0,
            0, 0, -1, 0,
            0, 0, -1, 0
        );

        assert_eq!(r, mem::transmute(__msa_ceqi_b(mem::transmute(a), -4)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ceqi_h() {
        #[rustfmt::skip]
        let a = i16x8::new(
            32767, 3276, 100, -11,
            32767, 3276, 100, -11
        );
        #[rustfmt::skip]
        let r = i16x8::new(0, 0, 0, -1, 0, 0, 0, -1);

        assert_eq!(r, mem::transmute(__msa_ceqi_h(mem::transmute(a), -11)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ceqi_w() {
        #[rustfmt::skip]
        let a = i32x4::new(1, 3, 5, -3);
        #[rustfmt::skip]
        let r = i32x4::new(0, 0, -1, 0);

        assert_eq!(r, mem::transmute(__msa_ceqi_w(mem::transmute(a), 5)));
    }

    // FIXME: https://reviews.llvm.org/D59884
    // If target type is i64, negative immediate loses the sign
    // Test passes if 4294967293 is used instead -3 in vector `a`
    // #[simd_test(enable = "msa")]
    // unsafe fn test_msa_ceqi_d() {
    //     #[rustfmt::skip]
    //     let a = i64x2::new(-3, 2);
    //     #[rustfmt::skip]
    //     let r = i64x2::new(-1, 0);

    //     assert_eq!(r, mem::transmute(__msa_ceqi_d(mem::transmute(a), -3)));
    // }

    // Can not be tested in user mode
    // #[simd_test(enable = "msa")]
    // unsafe fn test_msa_cfcmsa() {
    //     let r = 5;

    //     assert_eq!(r, mem::transmute(__msa_cfcmsa(5));
    // }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_cle_s_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            -128, 127, 55, 2,
            -128, 127, 55, 2,
            -128, 127, 55, 2,
            -128, 127, 55, 2
        );
        #[rustfmt::skip]
        let b = i8x16::new(
            -128, 126, 55, 1,
            -128, 126, 55, 1,
            -128, 126, 55, 1,
            -128, 126, 55, 1
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            -1, 0, -1, 0, 
            -1, 0, -1, 0, 
            -1, 0, -1, 0, 
            -1, 0, -1, 0
        );

        assert_eq!(
            r,
            mem::transmute(__msa_cle_s_b(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_cle_s_h() {
        #[rustfmt::skip]
        let a = i16x8::new(255, 155, 55, 2, 255, 155, 55, 2);
        #[rustfmt::skip]
        let b = i16x8::new(255, 155, 56, 1, 255, 155, 56, 1);
        #[rustfmt::skip]
        let r = i16x8::new(-1, -1, -1, 0, -1, -1, -1, 0);

        assert_eq!(
            r,
            mem::transmute(__msa_cle_s_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_cle_s_w() {
        #[rustfmt::skip]
        let a = i32x4::new(255, 155, 55, 2);
        #[rustfmt::skip]
        let b = i32x4::new(255, 156, 55, 1);
        #[rustfmt::skip]
        let r = i32x4::new(-1, -1, -1, 0);

        assert_eq!(
            r,
            mem::transmute(__msa_cle_s_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_cle_s_d() {
        #[rustfmt::skip]
        let a = i64x2::new(255, 155);
        #[rustfmt::skip]
        let b = i64x2::new(255, 156);
        #[rustfmt::skip]
        let r = i64x2::new(-1, -1);

        assert_eq!(
            r,
            mem::transmute(__msa_cle_s_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_cle_u_b() {
        #[rustfmt::skip]
        let a = u8x16::new(
            u8::MAX, 127, 55, 2,
            u8::MAX, 127, 55, 2,
            u8::MAX, 127, 55, 2,
            u8::MAX, 127, 55, 2
        );
        #[rustfmt::skip]
        let b = u8x16::new(
            u8::MAX, 126, 55, 1,
            u8::MAX, 126, 55, 1,
            u8::MAX, 126, 55, 1,
            u8::MAX, 126, 55, 1
        );
        #[rustfmt::skip]
        let r = i8x16::new(-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0);

        assert_eq!(
            r,
            mem::transmute(__msa_cle_u_b(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_cle_u_h() {
        #[rustfmt::skip]
        let a = u16x8::new(
            u16::MAX, 155, 55, 2, 
            u16::MAX, 155, 55, 2
        );
        #[rustfmt::skip]
        let b = u16x8::new(
            u16::MAX, 155, 56, 1, 
            u16::MAX, 155, 56, 1
        );
        #[rustfmt::skip]
        let r = i16x8::new(-1, -1, -1, 0, -1, -1, -1, 0);

        assert_eq!(
            r,
            mem::transmute(__msa_cle_u_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_cle_u_w() {
        #[rustfmt::skip]
        let a = u32x4::new(u32::MAX, 155, 55, 2);
        #[rustfmt::skip]
        let b = u32x4::new(u32::MAX, 156, 55, 1);
        #[rustfmt::skip]
        let r = i32x4::new(-1, -1, -1, 0);

        assert_eq!(
            r,
            mem::transmute(__msa_cle_u_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_cle_u_d() {
        #[rustfmt::skip]
        let a = u64x2::new(u64::MAX, 155);
        #[rustfmt::skip]
        let b = u64x2::new(u64::MAX, 156);
        #[rustfmt::skip]
        let r = i64x2::new(-1, -1);

        assert_eq!(
            r,
            mem::transmute(__msa_cle_u_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_clei_s_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            -2, -127, 100, -127,
            -2, -127, 100, -127,
            -2, -127, 100, -127,
            -2, -127, 100, -127
        );
        #[rustfmt::skip]
        let r = i8x16::new(-1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1);

        assert_eq!(r, mem::transmute(__msa_clei_s_b(mem::transmute(a), -2)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_clei_s_h() {
        #[rustfmt::skip]
        let a = i16x8::new(
            32767, 3276, 10, -1,
            32767, 3276, 10, -1,
        );
        #[rustfmt::skip]
        let r = i16x8::new(0, 0, 0, -1, 0, 0, 0, -1);

        assert_eq!(r, mem::transmute(__msa_clei_s_h(mem::transmute(a), -1)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_clei_s_w() {
        #[rustfmt::skip]
        let a = i32x4::new(100, 2147483647, 6, 2147483647);
        #[rustfmt::skip]
        let r = i32x4::new(0, 0, -1, 0);

        assert_eq!(r, mem::transmute(__msa_clei_s_w(mem::transmute(a), 6)));
    }

    // FIXME: https://reviews.llvm.org/D59884
    // If target type is i64, negative immediate loses the sign
    // -3 is represented as 4294967293
    // #[simd_test(enable = "msa")]
    // unsafe fn test_msa_clei_s_d() {
    //     #[rustfmt::skip]
    //     let a = i64x2::new(-3, 11);
    //     #[rustfmt::skip]
    //     let r = i64x2::new(-1, 0);

    //     assert_eq!(r, mem::transmute(__msa_clei_s_d(mem::transmute(a), -3)));
    // }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_clei_u_b() {
        #[rustfmt::skip]
        let a = u8x16::new(
            2, 127, 100, 127,
            2, 127, 100, 127,
            2, 127, 100, 127,
            2, 127, 100, 127,
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            -1, 0, 0, 0, 
            -1, 0, 0, 0, 
            -1, 0, 0, 0, 
            -1, 0, 0, 0
        );

        assert_eq!(r, mem::transmute(__msa_clei_u_b(mem::transmute(a), 25)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_clei_u_h() {
        #[rustfmt::skip]
        let a = u16x8::new(
            1, 26, 15, 36,
            1, 26, 15, 36
        );
        #[rustfmt::skip]
        let r = i16x8::new(-1, 0, -1, 0, -1, 0, -1, 0);

        assert_eq!(r, mem::transmute(__msa_clei_u_h(mem::transmute(a), 25)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_clei_u_w() {
        #[rustfmt::skip]
        let a = u32x4::new(25, 32, 25, 32);
        #[rustfmt::skip]
        let r = i32x4::new(-1, 0, -1, 0);

        assert_eq!(r, mem::transmute(__msa_clei_u_w(mem::transmute(a), 31)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_clei_u_d() {
        #[rustfmt::skip]
        let a = u64x2::new(10, 26);
        #[rustfmt::skip]
        let r = i64x2::new(-1, 0);

        assert_eq!(r, mem::transmute(__msa_clei_u_d(mem::transmute(a), 25)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_clt_s_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            -128, 127, 55, 2,
            -128, 127, 55, 2,
            -128, 127, 55, 2,
            -128, 127, 55, 2
        );
        #[rustfmt::skip]
        let b = i8x16::new(
            -127, 126, 56, 1,
            -127, 126, 56, 1,
            -127, 126, 56, 1,
            -127, 126, 56, 1
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            -1, 0, -1, 0, 
            -1, 0, -1, 0, 
            -1, 0, -1, 0, 
            -1, 0, -1, 0
        );

        assert_eq!(
            r,
            mem::transmute(__msa_clt_s_b(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_clt_s_h() {
        #[rustfmt::skip]
        let a = i16x8::new(-255, 155, 55, 2, -255, 155, 55, 2);
        #[rustfmt::skip]
        let b = i16x8::new(255, 156, 56, 1, 255, 156, 56, 1);
        #[rustfmt::skip]
        let r = i16x8::new(-1, -1, -1, 0, -1, -1, -1, 0);

        assert_eq!(
            r,
            mem::transmute(__msa_clt_s_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_clt_s_w() {
        #[rustfmt::skip]
        let a = i32x4::new(-255, 155, 55, 2);
        #[rustfmt::skip]
        let b = i32x4::new(255, 156, 55, 1);
        #[rustfmt::skip]
        let r = i32x4::new(-1, -1, 0, 0);

        assert_eq!(
            r,
            mem::transmute(__msa_clt_s_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_clt_s_d() {
        #[rustfmt::skip]
        let a = i64x2::new(-255, 155);
        #[rustfmt::skip]
        let b = i64x2::new(255, 156);
        #[rustfmt::skip]
        let r = i64x2::new(-1, -1);

        assert_eq!(
            r,
            mem::transmute(__msa_clt_s_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_clt_u_b() {
        #[rustfmt::skip]
        let a = u8x16::new(
            128, 127, 55, 2,
            128, 127, 55, 2,
            128, 127, 55, 2,
            128, 127, 55, 2
        );
        #[rustfmt::skip]
        let b = u8x16::new(
            127, 126, 56, 1,
            127, 126, 56, 1,
            127, 126, 56, 1,
            127, 126, 56, 1
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            0, 0, -1, 0, 
            0, 0, -1, 0, 
            0, 0, -1, 0, 
            0, 0, -1, 0
        );

        assert_eq!(
            r,
            mem::transmute(__msa_clt_u_b(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_clt_u_h() {
        #[rustfmt::skip]
        let a = u16x8::new(255, 155, 55, 2, 255, 155, 55, 2);
        #[rustfmt::skip]
        let b = u16x8::new(255, 156, 56, 1, 255, 156, 56, 1);
        #[rustfmt::skip]
        let r = i16x8::new(0, -1, -1, 0, 0, -1, -1, 0);

        assert_eq!(
            r,
            mem::transmute(__msa_clt_u_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_clt_u_w() {
        #[rustfmt::skip]
        let a = u32x4::new(255, 155, 55, 2);
        #[rustfmt::skip]
        let b = u32x4::new(255, 156, 55, 1);
        #[rustfmt::skip]
        let r = i32x4::new(0, -1, 0, 0);

        assert_eq!(
            r,
            mem::transmute(__msa_clt_u_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_clt_u_d() {
        #[rustfmt::skip]
        let a = u64x2::new(255, 155);
        #[rustfmt::skip]
        let b = u64x2::new(255, 156);
        #[rustfmt::skip]
        let r = i64x2::new(0, -1);

        assert_eq!(
            r,
            mem::transmute(__msa_clt_u_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_clti_s_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            2, -127, -5, 127,
            2, -127, -5, 127,
            2, -127, -5, 127,
            2, -127, -5, 127
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            0, -1, 0, 0, 
            0, -1, 0, 0, 
            0, -1, 0, 0, 
            0, -1, 0, 0
        );

        assert_eq!(r, mem::transmute(__msa_clti_s_b(mem::transmute(a), -5)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_clti_s_h() {
        #[rustfmt::skip]
        let a = i16x8::new(
            -1024, 3276, 15, 127,
            -1024, 3276, 15, 127
        );
        #[rustfmt::skip]
        let r = i16x8::new(-1, 0, 0, 0, -1, 0, 0, 0);

        assert_eq!(r, mem::transmute(__msa_clti_s_h(mem::transmute(a), 15)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_clti_s_w() {
        #[rustfmt::skip]
        let a = i32x4::new(-15, 2147483647, -15, 2147483647);
        #[rustfmt::skip]
        let r = i32x4::new(-1, 0, -1, 0);

        assert_eq!(r, mem::transmute(__msa_clti_s_w(mem::transmute(a), -10)));
    }

    // FIXME: https://reviews.llvm.org/D59884
    // If target type is i64, negative immediate loses the sign
    // -3 is represented as 4294967293
    // #[simd_test(enable = "msa")]
    // unsafe fn test_msa_clti_s_d() {
    //     #[rustfmt::skip]
    //     let a = i64x2::new(-5, -2);
    //     #[rustfmt::skip]
    //     let r = i64x2::new(-1, 0);

    //     assert_eq!(r, mem::transmute(__msa_clti_s_d(mem::transmute(a), -3)));
    // }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_clti_u_b() {
        #[rustfmt::skip]
        let a = u8x16::new(
            2, 127, 49, 127,
            2, 127, 49, 127,
            2, 127, 49, 127,
            2, 127, 49, 127,
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            -1, 0, 0, 0,
            -1, 0, 0, 0,
            -1, 0, 0, 0,
            -1, 0, 0, 0
        );

        assert_eq!(r, mem::transmute(__msa_clti_u_b(mem::transmute(a), 50)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_clti_u_h() {
        #[rustfmt::skip]
        let a = u16x8::new(
            327, 3276, 100, 127,
            327, 3276, 100, 127
        );
        #[rustfmt::skip]
        let r = i16x8::new(0, 0, 0, 0, 0, 0, 0, 0);

        assert_eq!(r, mem::transmute(__msa_clti_u_h(mem::transmute(a), 30)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_clti_u_w() {
        #[rustfmt::skip]
        let a = u32x4::new(100, 2147483647, 100, 2147483647);
        #[rustfmt::skip]
        let r = i32x4::new(0, 0, 0, 0);

        assert_eq!(r, mem::transmute(__msa_clti_u_w(mem::transmute(a), 10)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_clti_u_d() {
        #[rustfmt::skip]
        let a = u64x2::new(1, 9223372036854775807);
        #[rustfmt::skip]
        let r = i64x2::new(-1, 0);

        assert_eq!(r, mem::transmute(__msa_clti_u_d(mem::transmute(a), 10)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_copy_s_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            -100, 127, 4, 127,
            -100, 127, 4, 127,
            -100, 127, 4, 127,
            -100, 127, 4, 127
        );
        #[rustfmt::skip]
        let r = -100 as i32;

        assert_eq!(r, mem::transmute(__msa_copy_s_b(mem::transmute(a), 12)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_copy_s_h() {
        #[rustfmt::skip]
        let a = i16x8::new(
            32767, 3276, 100, 11,
            32767, 3276, 100, 11
        );
        #[rustfmt::skip]
        let r = 32767 as i32;

        assert_eq!(r, mem::transmute(__msa_copy_s_h(mem::transmute(a), 4)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_copy_s_w() {
        #[rustfmt::skip]
        let a = i32x4::new(100, 2147483647, 5, -2147483647);
        let r = 2147483647 as i32;

        assert_eq!(r, mem::transmute(__msa_copy_s_w(mem::transmute(a), 1)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_copy_s_d() {
        #[rustfmt::skip]
        let a = i64x2::new(3, 9223372036854775807);
        #[rustfmt::skip]
        let r = 9223372036854775807 as i64;

        assert_eq!(r, mem::transmute(__msa_copy_s_d(mem::transmute(a), 1)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_copy_u_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            100, 127, 4, 127,
            100, 127, 4, 127,
            100, 127, 4, 127,
            100, 127, 4, 127
        );
        #[rustfmt::skip]
        let r = 100 as u32;

        assert_eq!(r, mem::transmute(__msa_copy_u_b(mem::transmute(a), 12)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_copy_u_h() {
        #[rustfmt::skip]
        let a = i16x8::new(
            32767, 3276, 100, 11,
            32767, 3276, 100, 11
        );
        #[rustfmt::skip]
        let r = 32767 as u32;

        assert_eq!(r, mem::transmute(__msa_copy_u_h(mem::transmute(a), 4)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_copy_u_w() {
        #[rustfmt::skip]
        let a = i32x4::new(100, 2147483647, 5, 2147483647);
        #[rustfmt::skip]
        let r = 2147483647 as u32;

        assert_eq!(r, mem::transmute(__msa_copy_u_w(mem::transmute(a), 1)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_copy_u_d() {
        #[rustfmt::skip]
        let a = i64x2::new(3, i64::MAX);
        #[rustfmt::skip]
        let r = 9223372036854775807 as u64;

        assert_eq!(r, mem::transmute(__msa_copy_u_d(mem::transmute(a), 1)));
    }

    // Can not be tested in user mode
    // #[simd_test(enable = "msa")]
    // unsafe fn test_msa_ctcmsa() {
    // }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_div_s_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            -6, -7, -8, -9,
            -6, -7, -8, -9,
            -6, -7, -8, -9,
            -6, -7, -8, -9
        );
        #[rustfmt::skip]
        let b = i8x16::new(
            -1, -2, -3, -4,
            -1, -2, -3, -4,
            -1, -2, -3, -4,
            -1, -2, -3, -4
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            6, 3, 2, 2, 
            6, 3, 2, 2, 
            6, 3, 2, 2, 
            6, 3, 2, 2
        );

        assert_eq!(
            r,
            mem::transmute(__msa_div_s_b(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_div_s_h() {
        #[rustfmt::skip]
        let a = i16x8::new(-6, -7, -8, -9, 6, 7, 8, 9);
        #[rustfmt::skip]
        let b = i16x8::new(-1, -2, -3, -4, -1, -2, -3, -4);
        #[rustfmt::skip]
        let r = i16x8::new(6, 3, 2, 2, -6, -3, -2, -2);

        assert_eq!(
            r,
            mem::transmute(__msa_div_s_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_div_s_w() {
        #[rustfmt::skip]
        let a = i32x4::new(-6, -7, 8, 9);
        #[rustfmt::skip]
        let b = i32x4::new(-1, -2, -3, -4);
        #[rustfmt::skip]
        let r = i32x4::new(6, 3, -2, -2);

        assert_eq!(
            r,
            mem::transmute(__msa_div_s_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_div_s_d() {
        #[rustfmt::skip]
        let a = i64x2::new(-6, 7);
        #[rustfmt::skip]
        let b = i64x2::new(-1, -2);
        #[rustfmt::skip]
        let r = i64x2::new(6, -3);

        assert_eq!(
            r,
            mem::transmute(__msa_div_s_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_div_u_b() {
        #[rustfmt::skip]
        let a = u8x16::new(
            6, 7, 8, 9,
            6, 7, 8, 9,
            6, 7, 8, 9,
            6, 7, 8, 9
        );
        #[rustfmt::skip]
        let b = u8x16::new(
            1, 2, 3, 4,
            1, 2, 3, 4,
            1, 2, 3, 4,
            1, 2, 3, 4
        );
        #[rustfmt::skip]
        let r = u8x16::new(
            6, 3, 2, 2, 
            6, 3, 2, 2, 
            6, 3, 2, 2, 
            6, 3, 2, 2
        );

        assert_eq!(
            r,
            mem::transmute(__msa_div_u_b(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_div_u_h() {
        #[rustfmt::skip]
        let a = u16x8::new(6, 7, 8, 9, 6, 7, 8, 9);
        #[rustfmt::skip]
        let b = u16x8::new(1, 2, 3, 4, 1, 2, 3, 4);
        #[rustfmt::skip]
        let r = u16x8::new(6, 3, 2, 2, 6, 3, 2, 2);

        assert_eq!(
            r,
            mem::transmute(__msa_div_u_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_div_u_w() {
        #[rustfmt::skip]
        let a = u32x4::new(6, 7, 8, 9);
        #[rustfmt::skip]
        let b = u32x4::new(1, 2, 3, 4);
        #[rustfmt::skip]
        let r = u32x4::new(6, 3, 2, 2);

        assert_eq!(
            r,
            mem::transmute(__msa_div_u_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_div_u_d() {
        #[rustfmt::skip]
        let a = u64x2::new(6, 7);
        #[rustfmt::skip]
        let b = u64x2::new(1, 2);
        #[rustfmt::skip]
        let r = u64x2::new(6, 3);

        assert_eq!(
            r,
            mem::transmute(__msa_div_u_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_dotp_s_h() {
        #[rustfmt::skip]
        let a = i8x16::new(
            -1, -2, -3, 4,
            -1, -2, -3, -4,
            -1, -2, -3, 4,
            -1, -2, -3, -4
        );
        #[rustfmt::skip]
        let b = i8x16::new(
            -6, -7, -8, -9,
            -6, -7, -8, -9,
            -6, -7, -8, -9,
            -6, -7, -8, -9
        );
        #[rustfmt::skip]
        let r = i16x8::new(20, -12, 20, 60, 20, -12, 20, 60);

        assert_eq!(
            r,
            mem::transmute(__msa_dotp_s_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_dotp_s_w() {
        #[rustfmt::skip]
        let a = i16x8::new(-1, -2, -3, -4, -1, -2, -3, 4);
        #[rustfmt::skip]
        let b = i16x8::new(-6, -7, -8, -9, -6, -7, -8, -9);
        #[rustfmt::skip]
        let r = i32x4::new(20, 60, 20, -12);

        assert_eq!(
            r,
            mem::transmute(__msa_dotp_s_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_dotp_s_d() {
        #[rustfmt::skip]
        let a = i32x4::new(-1, -2, -3, 4);
        #[rustfmt::skip]
        let b = i32x4::new(-6, -7, -8, -9);
        #[rustfmt::skip]
        let r = i64x2::new(20, -12);

        assert_eq!(
            r,
            mem::transmute(__msa_dotp_s_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_dotp_u_h() {
        #[rustfmt::skip]
        let a = u8x16::new(
            1, 2, 3, 4,
            1, 2, 3, 4,
            1, 2, 3, 4,
            1, 2, 3, 4
        );
        #[rustfmt::skip]
        let b = u8x16::new(
            6, 7, 8, 9,
            6, 7, 8, 9,
            6, 7, 8, 9,
            6, 7, 8, 9
        );
        #[rustfmt::skip]
        let r = u16x8::new(20, 60, 20, 60, 20, 60, 20, 60);

        assert_eq!(
            r,
            mem::transmute(__msa_dotp_u_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_dotp_u_w() {
        #[rustfmt::skip]
        let a = u16x8::new(1, 2, 3, 4, 1, 2, 3, 4);
        #[rustfmt::skip]
        let b = u16x8::new(6, 7, 8, 9, 6, 7, 8, 9);
        #[rustfmt::skip]
        let r = u32x4::new(20, 60, 20, 60);

        assert_eq!(
            r,
            mem::transmute(__msa_dotp_u_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_dotp_u_d() {
        #[rustfmt::skip]
        let a = u32x4::new(1, 2, 3, 4);
        #[rustfmt::skip]
        let b = u32x4::new(6, 7, 8, 9);
        #[rustfmt::skip]
        let r = u64x2::new(20, 60);

        assert_eq!(
            r,
            mem::transmute(__msa_dotp_u_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_dpadd_s_h() {
        #[rustfmt::skip]
        let a = i16x8::new(-1, -2, -3, -4, -1, -2, -3, 4);
        #[rustfmt::skip]
        let b = i8x16::new(
            -1, -2, -3, 4,
            -1, -2, -3, -4,
            -1, -2, -3, 4,
            -1, -2, -3, -4
        );
        #[rustfmt::skip]
        let c = i8x16::new(
            -6, -7, -8, -9,
            -6, -7, -8, -9,
            -6, -7, -8, -9,
            -6, -7, -8, -9
        );
        #[rustfmt::skip]
        let r = i16x8::new(19, -14, 17, 56, 19, -14, 17, 64);

        assert_eq!(
            r,
            mem::transmute(__msa_dpadd_s_h(
                mem::transmute(a),
                mem::transmute(b),
                mem::transmute(c)
            ))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_dpadd_s_w() {
        #[rustfmt::skip]
        let a = i32x4::new(-1, -2, -3, -4);
        #[rustfmt::skip]
        let b = i16x8::new(
            -1, -2, -3, 4,
            -1, -2, -3, -4
        );
        #[rustfmt::skip]
        let c = i16x8::new(
            -6, -7, -8, -9,
            -6, -7, -8, -9
        );
        #[rustfmt::skip]
        let r = i32x4::new(19, -14, 17, 56);

        assert_eq!(
            r,
            mem::transmute(__msa_dpadd_s_w(
                mem::transmute(a),
                mem::transmute(b),
                mem::transmute(c)
            ))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_dpadd_s_d() {
        #[rustfmt::skip]
        let a = i64x2::new(-1, -2);
        #[rustfmt::skip]
        let b = i32x4::new(-1, -2, -3, 4);
        #[rustfmt::skip]
        let c = i32x4::new(-6, -7, -8, -9);
        #[rustfmt::skip]
        let r = i64x2::new(19, -14);

        assert_eq!(
            r,
            mem::transmute(__msa_dpadd_s_d(
                mem::transmute(a),
                mem::transmute(b),
                mem::transmute(c)
            ))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_dpadd_u_h() {
        #[rustfmt::skip]
        let a = u16x8::new(1, 2, 3, 4, 1, 2, 3, 4);
        #[rustfmt::skip]
        let b = u8x16::new(
            1, 2, 3, 4,
            1, 2, 3, 4,
            1, 2, 3, 4,
            1, 2, 3, 4
        );
        #[rustfmt::skip]
        let c = u8x16::new(
            6, 7, 8, 9,
            6, 7, 8, 9,
            6, 7, 8, 9,
            6, 7, 8, 9
        );
        #[rustfmt::skip]
        let r = u16x8::new(21, 62, 23, 64, 21, 62, 23, 64);

        assert_eq!(
            r,
            mem::transmute(__msa_dpadd_u_h(
                mem::transmute(a),
                mem::transmute(b),
                mem::transmute(c)
            ))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_dpadd_u_w() {
        #[rustfmt::skip]
        let a = u32x4::new(1, 2, 3, 4);
        #[rustfmt::skip]
        let b = u16x8::new(
            1, 2, 3, 4,
            1, 2, 3, 4
        );
        #[rustfmt::skip]
        let c = u16x8::new(
            6, 7, 8, 9,
            6, 7, 8, 9
        );
        #[rustfmt::skip]
        let r = u32x4::new(21, 62, 23, 64);

        assert_eq!(
            r,
            mem::transmute(__msa_dpadd_u_w(
                mem::transmute(a),
                mem::transmute(b),
                mem::transmute(c)
            ))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_dpadd_u_d() {
        #[rustfmt::skip]
        let a = u64x2::new(1, 2);
        #[rustfmt::skip]
        let b = u32x4::new(1, 2, 3, 4);
        #[rustfmt::skip]
        let c = u32x4::new(6, 7, 8, 9);
        #[rustfmt::skip]
        let r = u64x2::new(21, 62);

        assert_eq!(
            r,
            mem::transmute(__msa_dpadd_u_d(
                mem::transmute(a),
                mem::transmute(b),
                mem::transmute(c)
            ))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_dpsub_s_h() {
        #[rustfmt::skip]
        let a = i16x8::new(-1, -2, -3, -4, -1, -2, -3, 4);
        #[rustfmt::skip]
        let b = i8x16::new(
            -1, -2, -3, 4,
            -1, -2, -3, -4,
            -1, -2, -3, 4,
            -1, -2, -3, -4
        );
        #[rustfmt::skip]
        let c = i8x16::new(
            -6, -7, -8, -9,
            -6, -7, -8, -9,
            -6, -7, -8, -9,
            -6, -7, -8, -9
        );
        #[rustfmt::skip]
        let r = i16x8::new(-21, 10, -23, -64, -21, 10, -23, -56);

        assert_eq!(
            r,
            mem::transmute(__msa_dpsub_s_h(
                mem::transmute(a),
                mem::transmute(b),
                mem::transmute(c)
            ))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_dpsub_s_w() {
        #[rustfmt::skip]
        let a = i32x4::new(-1, -2, -3, -4);
        #[rustfmt::skip]
        let b = i16x8::new(
            -1, -2, -3, 4,
            -1, -2, -3, -4
        );
        #[rustfmt::skip]
        let c = i16x8::new(
            -6, -7, -8, -9,
            -6, -7, -8, -9
        );
        #[rustfmt::skip]
        let r = i32x4::new(-21, 10, -23, -64);

        assert_eq!(
            r,
            mem::transmute(__msa_dpsub_s_w(
                mem::transmute(a),
                mem::transmute(b),
                mem::transmute(c)
            ))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_dpsub_s_d() {
        #[rustfmt::skip]
        let a = i64x2::new(-1, -2);
        #[rustfmt::skip]
        let b = i32x4::new(-1, -2, -3, 4);
        #[rustfmt::skip]
        let c = i32x4::new(-6, -7, -8, -9);
        #[rustfmt::skip]
        let r = i64x2::new(-21, 10);

        assert_eq!(
            r,
            mem::transmute(__msa_dpsub_s_d(
                mem::transmute(a),
                mem::transmute(b),
                mem::transmute(c)
            ))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_dpsub_u_h() {
        #[rustfmt::skip]
        let a = i16x8::new(1, -2, 3, -4, -1, 2,-3, 4);
        #[rustfmt::skip]
        let b = u8x16::new(
            1, 2, 3, 4,
            1, 2, 3, 4,
            1, 2, 3, 4,
            1, 2, 3, 4        
        );
        #[rustfmt::skip]
        let c = u8x16::new(
            6, 7, 8, 9,
            6, 7, 8, 9,
            6, 7, 8, 9,
            6, 7, 8, 9       
        );
        #[rustfmt::skip]
        let r = i16x8::new(-19, -62, -17, -64, -21, -58, -23, -56);

        assert_eq!(
            r,
            mem::transmute(__msa_dpsub_u_h(
                mem::transmute(a),
                mem::transmute(b),
                mem::transmute(c)
            ))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_dpsub_u_w() {
        #[rustfmt::skip]
        let a = i32x4::new(1, -2, 3, -4);
        #[rustfmt::skip]
        let b = u16x8::new(
            1, 2, 3, 4,
            1, 2, 3, 4
        );
        #[rustfmt::skip]
        let c = u16x8::new(
            6, 7, 8, 9,
            6, 7, 8, 9
        );
        #[rustfmt::skip]
        let r = i32x4::new(-19, -62, -17, -64);

        assert_eq!(
            r,
            mem::transmute(__msa_dpsub_u_w(
                mem::transmute(a),
                mem::transmute(b),
                mem::transmute(c)
            ))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_dpsub_u_d() {
        #[rustfmt::skip]
        let a = i64x2::new(1, -2);
        #[rustfmt::skip]
        let b = u32x4::new(1, 2, 3, 4);
        #[rustfmt::skip]
        let c = u32x4::new(6, 7, 8, 9);
        #[rustfmt::skip]
        let r = i64x2::new(-19, -62);

        assert_eq!(
            r,
            mem::transmute(__msa_dpsub_u_d(
                mem::transmute(a),
                mem::transmute(b),
                mem::transmute(c)
            ))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fadd_w() {
        #[rustfmt::skip]
        let a = f32x4::new(1.1, -2.2, 3.3, -4.4);
        #[rustfmt::skip]
        let b = f32x4::new(4.4, -3.3, 2.2, -1.1);
        #[rustfmt::skip]
        let r = f32x4::new(5.5, -5.5, 5.5, -5.5);

        assert_eq!(
            r,
            mem::transmute(__msa_fadd_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fadd_d() {
        #[rustfmt::skip]
        let a = f64x2::new(1.1, -2.2);
        #[rustfmt::skip]
        let b = f64x2::new(4.4, -3.3);
        #[rustfmt::skip]
        let r = f64x2::new(5.5, -5.5);

        assert_eq!(
            r,
            mem::transmute(__msa_fadd_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    // Only observed beahiour should be SIGFPE signal
    // Can not be tested
    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fcaf_w() {
        #[rustfmt::skip]
        let a = f32x4::new(1.1, -2.2, 3.3, -4.4);
        #[rustfmt::skip]
        let b = f32x4::new(0.0, -1.2, 3.3, f32::NAN);
        #[rustfmt::skip]
        let r = i32x4::new(0, 0, 0, 0);

        assert_eq!(
            r,
            mem::transmute(__msa_fcaf_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    // Only observed beahiour should be SIGFPE signal
    // Can not be tested
    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fcaf_d() {
        #[rustfmt::skip]
        let a = f64x2::new(1.1, -2.2);
        #[rustfmt::skip]
        let b = f64x2::new(-2.2, 1.1);
        #[rustfmt::skip]
        let r = i64x2::new(0, 0);

        assert_eq!(
            r,
            mem::transmute(__msa_fcaf_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fceq_w() {
        #[rustfmt::skip]
        let a = f32x4::new(1.1, -2.2, 3.3, f32::NAN);
        #[rustfmt::skip]
        let b = f32x4::new(-4.4, -2.2, 3.3, f32::NAN);
        #[rustfmt::skip]
        let r = i32x4::new(0, -1, -1, 0);

        assert_eq!(
            r,
            mem::transmute(__msa_fceq_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fceq_d() {
        #[rustfmt::skip]
        let a = f64x2::new(1.1, -2.2);
        #[rustfmt::skip]
        let b = f64x2::new(1.1, 1.1);
        #[rustfmt::skip]
        let r = i64x2::new(-1, 0);

        assert_eq!(
            r,
            mem::transmute(__msa_fceq_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fclass_w() {
        #[rustfmt::skip]
        let a = f32x4::new(1.1, -2.2, 3.3, f32::NAN);
        #[rustfmt::skip]
        let r = i32x4::new(128, 8, 128, 2);

        assert_eq!(r, mem::transmute(__msa_fclass_w(mem::transmute(a))));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fclass_d() {
        #[rustfmt::skip]
        let a = f64x2::new(1.1, -2.2);
        #[rustfmt::skip]
        let r = i64x2::new(128, 8);

        assert_eq!(r, mem::transmute(__msa_fclass_d(mem::transmute(a))));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fcle_w() {
        #[rustfmt::skip]
        let a = f32x4::new(1.1, -2.2, 3.3, f32::NAN);
        #[rustfmt::skip]
        let b = f32x4::new(-4.4, -1.2, 3.3, f32::NAN);
        #[rustfmt::skip]
        let r = i32x4::new(0, -1, -1, 0);

        assert_eq!(
            r,
            mem::transmute(__msa_fcle_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fcle_d() {
        #[rustfmt::skip]
        let a = f64x2::new(1.1, -2.2);
        #[rustfmt::skip]
        let b = f64x2::new(1.1, 1.1);
        #[rustfmt::skip]
        let r = i64x2::new(-1, -1);

        assert_eq!(
            r,
            mem::transmute(__msa_fcle_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fclt_w() {
        #[rustfmt::skip]
        let a = f32x4::new(1.1, -2.2, 3.3, f32::NAN);
        #[rustfmt::skip]
        let b = f32x4::new(-4.4, -1.2, 3.3, f32::NAN);
        #[rustfmt::skip]
        let r = i32x4::new(0, -1, 0, 0);

        assert_eq!(
            r,
            mem::transmute(__msa_fclt_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fclt_d() {
        #[rustfmt::skip]
        let a = f64x2::new(1.1, -2.2);
        #[rustfmt::skip]
        let b = f64x2::new(1.1, 1.1);
        #[rustfmt::skip]
        let r = i64x2::new(0, -1);

        assert_eq!(
            r,
            mem::transmute(__msa_fclt_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fcne_w() {
        #[rustfmt::skip]
        let a = f32x4::new(1.1, -2.2, 3.3, f32::NAN);
        #[rustfmt::skip]
        let b = f32x4::new(-4.4, -1.2, 3.3, f32::NAN);
        #[rustfmt::skip]
        let r = i32x4::new(-1, -1, 0, 0);

        assert_eq!(
            r,
            mem::transmute(__msa_fcne_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fcne_d() {
        #[rustfmt::skip]
        let a = f64x2::new(1.1, -2.2);
        #[rustfmt::skip]
        let b = f64x2::new(1.1, 1.1);
        #[rustfmt::skip]
        let r = i64x2::new(0, -1);

        assert_eq!(
            r,
            mem::transmute(__msa_fcne_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fcor_w() {
        #[rustfmt::skip]
        let a = f32x4::new(1.1, -2.2, 3.3, f32::NAN);
        #[rustfmt::skip]
        let b = f32x4::new(f32::NAN, -1.2, 3.3, f32::NAN);
        #[rustfmt::skip]
        let r = i32x4::new(0, -1, -1, 0);

        assert_eq!(
            r,
            mem::transmute(__msa_fcor_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fcor_d() {
        #[rustfmt::skip]
        let a = f64x2::new(1.1, f64::NAN);
        #[rustfmt::skip]
        let b = f64x2::new(1.1, 1.1);
        #[rustfmt::skip]
        let r = i64x2::new(-1, 0);

        assert_eq!(
            r,
            mem::transmute(__msa_fcor_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fcueq_w() {
        #[rustfmt::skip]
        let a = f32x4::new(1.1, -2.2, 3.3, f32::NAN);
        #[rustfmt::skip]
        let b = f32x4::new(f32::NAN, -1.2, 3.3, f32::NAN);
        #[rustfmt::skip]
        let r = i32x4::new(-1, 0, -1, -1);

        assert_eq!(
            r,
            mem::transmute(__msa_fcueq_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fcueq_d() {
        #[rustfmt::skip]
        let a = f64x2::new(1.1, f64::NAN);
        #[rustfmt::skip]
        let b = f64x2::new(1.1, 1.1);
        #[rustfmt::skip]
        let r = i64x2::new(-1, -1);

        assert_eq!(
            r,
            mem::transmute(__msa_fcueq_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fcule_w() {
        #[rustfmt::skip]
        let a = f32x4::new(1.1, -2.2, 3.3, f32::NAN);
        #[rustfmt::skip]
        let b = f32x4::new(f32::NAN, -1.2, 3.3, f32::NAN);
        #[rustfmt::skip]
        let r = i32x4::new(-1, -1, -1, -1);

        assert_eq!(
            r,
            mem::transmute(__msa_fcule_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fcule_d() {
        #[rustfmt::skip]
        let a = f64x2::new(1.1, f64::NAN);
        #[rustfmt::skip]
        let b = f64x2::new(1.1, 1.1);
        #[rustfmt::skip]
        let r = i64x2::new(-1, -1);

        assert_eq!(
            r,
            mem::transmute(__msa_fcule_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fcult_w() {
        #[rustfmt::skip]
        let a = f32x4::new(1.1, -2.2, 3.3, f32::NAN);
        #[rustfmt::skip]
        let b = f32x4::new(f32::NAN, -1.2, 3.3, f32::NAN);
        #[rustfmt::skip]
        let r = i32x4::new(-1, -1, 0, -1);

        assert_eq!(
            r,
            mem::transmute(__msa_fcult_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fcult_d() {
        #[rustfmt::skip]
        let a = f64x2::new(1.1, f64::NAN);
        #[rustfmt::skip]
        let b = f64x2::new(1.1, 1.1);
        #[rustfmt::skip]
        let r = i64x2::new(0, -1);

        assert_eq!(
            r,
            mem::transmute(__msa_fcult_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fcun_w() {
        #[rustfmt::skip]
        let a = f32x4::new(1.1, -2.2, 3.3, f32::NAN);
        #[rustfmt::skip]
        let b = f32x4::new(f32::NAN, -1.2, 3.3, f32::NAN);
        #[rustfmt::skip]
        let r = i32x4::new(-1, 0, 0, -1);

        assert_eq!(
            r,
            mem::transmute(__msa_fcun_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fcun_d() {
        #[rustfmt::skip]
        let a = f64x2::new(1.1, f64::NAN);
        #[rustfmt::skip]
        let b = f64x2::new(1.1, 1.1);
        #[rustfmt::skip]
        let r = i64x2::new(0, -1);

        assert_eq!(
            r,
            mem::transmute(__msa_fcun_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fcune_w() {
        #[rustfmt::skip]
        let a = f32x4::new(1.1, -2.2, 3.3, f32::NAN);
        #[rustfmt::skip]
        let b = f32x4::new(f32::NAN, -1.2, 3.3, f32::NAN);
        #[rustfmt::skip]
        let r = i32x4::new(-1, -1, 0, -1);

        assert_eq!(
            r,
            mem::transmute(__msa_fcune_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fcune_d() {
        #[rustfmt::skip]
        let a = f64x2::new(1.1, f64::NAN);
        #[rustfmt::skip]
        let b = f64x2::new(1.1, 1.1);
        #[rustfmt::skip]
        let r = i64x2::new(0, -1);

        assert_eq!(
            r,
            mem::transmute(__msa_fcune_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fdiv_w() {
        #[rustfmt::skip]
        let a = f32x4::new(5.25, -20.2, 333.333, -425.0);
        #[rustfmt::skip]
        let b = f32x4::new(4.0, -2.1, 11.11, 8.2);
        #[rustfmt::skip]
        let r = f32x4::new(1.3125, 9.619048, 30.002972, -51.82927);

        assert_eq!(
            r,
            mem::transmute(__msa_fdiv_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fdiv_d() {
        #[rustfmt::skip]
        let a = f64x2::new(1111.11, -222222.2);
        #[rustfmt::skip]
        let b = f64x2::new(-4.85, 3.33);
        #[rustfmt::skip]
        let r = f64x2::new(-229.09484536082473, -66733.3933933934);

        assert_eq!(
            r,
            mem::transmute(__msa_fdiv_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    /*// FIXME: 16-bit floats
    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fexdo_h() {
        #[rustfmt::skip]
        let a = f32x4::new(20.5, 2.3, 4.5, 5.4);
        #[rustfmt::skip]
        let b = f32x4::new(1.1, 1.0, 1.0, 1.0);
        let r = i16x8::new(1, 9, 30, 51, 1, 9, 30, 51);

        assert_eq!(r, mem::transmute(__msa_fexdo_h(mem::transmute(a), mem::transmute(b))));
    }*/

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fexdo_w() {
        #[rustfmt::skip]
        let a = f64x2::new(2000005.5, 2.3);
        #[rustfmt::skip]
        let b = f64x2::new(1235689784512.1, 2147483649998.5);
        #[rustfmt::skip]
        let r = f32x4::new(
            1235689800000.0, 2147483600000.0, 
            2000005.5, 2.3
        );

        assert_eq!(
            r,
            mem::transmute(__msa_fexdo_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fexp2_w() {
        #[rustfmt::skip]
        let a = f32x4::new(1.1, -2.2, 3.3, -4.4);
        #[rustfmt::skip]
        let b = i32x4::new(4, -3, 2, 1);
        #[rustfmt::skip]
        let r = f32x4::new(17.6, -0.275, 13.2, -8.8);

        assert_eq!(
            r,
            mem::transmute(__msa_fexp2_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fexp2_d() {
        #[rustfmt::skip]
        let a = f64x2::new(1.1, -2.2);
        #[rustfmt::skip]
        let b = i64x2::new(-4, 3);
        #[rustfmt::skip]
        let r = f64x2::new(0.06875, -17.6);

        assert_eq!(
            r,
            mem::transmute(__msa_fexp2_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    // FIXME: 16-bit floats
    // #[simd_test(enable = "msa")]
    // unsafe fn test_msa_fexupl_w() {
    //     #[rustfmt::skip]
    //     let a = f16x8(1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5);
    //     #[rustfmt::skip]
    //     let r = f32x4::new(5.5, 6.5, 7.5, 8.5);

    //     assert_eq!(r, mem::transmute(__msa_fexupl_w(mem::transmute(a))));
    // }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fexupl_d() {
        #[rustfmt::skip]
        let a = f32x4::new(5.5, 6.5, 7.5, 8.5);
        #[rustfmt::skip]
        let r = f64x2::new(7.5, 8.5);

        assert_eq!(r, mem::transmute(__msa_fexupl_d(mem::transmute(a))));
    }

    // FIXME: 16-bit floats
    //     #[simd_test(enable = "msa")]
    // unsafe fn test_msa_fexupr_w() {
    //     #[rustfmt::skip]
    //     let a = f16x8(1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5);
    //     #[rustfmt::skip]
    //     let r = f32x4::new(1.5, 2.5, 3.5, 4.5);

    //     assert_eq!(r, mem::transmute(__msa_fexupr_w(mem::transmute(a))));
    // }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fexupr_d() {
        #[rustfmt::skip]
        let a = f32x4::new(5.5, 6.5, 7.5, 8.5);
        #[rustfmt::skip]
        let r = f64x2::new(5.5, 6.5);

        assert_eq!(r, mem::transmute(__msa_fexupr_d(mem::transmute(a))));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ffint_s_w() {
        #[rustfmt::skip]
        let a = i32x4::new(-1, 2, -3, 4);
        #[rustfmt::skip]
        let r = f32x4::new(-1.0, 2.0, -3.0, 4.0);

        assert_eq!(r, mem::transmute(__msa_ffint_s_w(mem::transmute(a))));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ffint_s_d() {
        #[rustfmt::skip]
        let a = i64x2::new(-1, 2);
        #[rustfmt::skip]
        let r = f64x2::new(-1.0,     2.0);

        assert_eq!(r, mem::transmute(__msa_ffint_s_d(mem::transmute(a))));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ffint_u_w() {
        #[rustfmt::skip]
        let a = u32x4::new(1, 2, 3, 4);
        #[rustfmt::skip]
        let r = f32x4::new(1.0, 2.0, 3.0, 4.0);

        assert_eq!(r, mem::transmute(__msa_ffint_u_w(mem::transmute(a))));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ffint_u_d() {
        #[rustfmt::skip]
        let a = u64x2::new(1, 2);
        #[rustfmt::skip]
        let r = f64x2::new(1.0, 2.0);

        assert_eq!(r, mem::transmute(__msa_ffint_u_d(mem::transmute(a))));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ffql_w() {
        #[rustfmt::skip]
        let a = i16x8::new(11, 25, 33, 47, 11, 25, 33, 47);
        #[rustfmt::skip]
        let r = f32x4::new(
            0.00033569336, 0.00076293945,
            0.0010070801, 0.0014343262
        );

        assert_eq!(r, mem::transmute(__msa_ffql_w(mem::transmute(a))));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ffql_d() {
        #[rustfmt::skip]
        let a = i32x4::new(1111, 2222, 3333, 4444);
        #[rustfmt::skip]
        let r = f64x2::new(
            0.000001552049070596695,
            0.0000020693987607955933
        );

        assert_eq!(r, mem::transmute(__msa_ffql_d(mem::transmute(a))));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ffqr_w() {
        #[rustfmt::skip]
        let a = i16x8::new(12, 26, 34, 48, 11, 25, 33, 47);
        #[rustfmt::skip]
        let r = f32x4::new(
            0.00036621094, 0.00079345703, 
            0.0010375977, 0.0014648438
        );

        assert_eq!(r, mem::transmute(__msa_ffqr_w(mem::transmute(a))));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ffqr_d() {
        #[rustfmt::skip]
        let a = i32x4::new(1111, 2555, 3333, 475);
        #[rustfmt::skip]
        let r = f64x2::new(
            0.0000005173496901988983, 
            0.0000011897645890712738
        );

        assert_eq!(r, mem::transmute(__msa_ffqr_d(mem::transmute(a))));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fill_b() {
        #[rustfmt::skip]
        let r = i8x16::new(
            2, 2, 2, 2,
            2, 2, 2, 2,
            2, 2, 2, 2,
            2, 2, 2, 2
        );

        assert_eq!(r, mem::transmute(__msa_fill_b(2)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fill_h() {
        #[rustfmt::skip]
        let r = i16x8::new(2, 2, 2, 2, 2, 2, 2, 2);

        assert_eq!(r, mem::transmute(__msa_fill_h(2)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fill_w() {
        #[rustfmt::skip]
        let r = i32x4::new(2, 2, 2, 2);

        assert_eq!(r, mem::transmute(__msa_fill_w(2)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fill_d() {
        #[rustfmt::skip]
        let r = i64x2::new(2, 2);

        assert_eq!(r, mem::transmute(__msa_fill_d(2)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_flog2_w() {
        #[rustfmt::skip]
        let a = f32x4::new(8.0, 16.0, 32.0, 64.0);
        #[rustfmt::skip]
        let r = f32x4::new(3.0, 4.0, 5.0, 6.0);

        assert_eq!(r, mem::transmute(__msa_flog2_w(mem::transmute(a))));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_flog2_d() {
        #[rustfmt::skip]
        let a = f64x2::new(8.0, 16.0);
        #[rustfmt::skip]
        let r = f64x2::new(3.0, 4.0);

        assert_eq!(r, mem::transmute(__msa_flog2_d(mem::transmute(a))));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fmadd_w() {
        #[rustfmt::skip]
        let a = f32x4::new(1.0, 2.0, 3.0, 4.0);
        #[rustfmt::skip]
        let b = f32x4::new(5.0, 6.0, 7.0, 8.0);
        #[rustfmt::skip]
        let c = f32x4::new(9.0, 10.0, 11.0, 12.0);
        #[rustfmt::skip]
        let r = f32x4::new(46.0, 62.0, 80.0, 100.0);

        assert_eq!(
            r,
            mem::transmute(__msa_fmadd_w(
                mem::transmute(a),
                mem::transmute(b),
                mem::transmute(c)
            ))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fmadd_d() {
        #[rustfmt::skip]
        let a = f64x2::new(1.0, 2.0);
        #[rustfmt::skip]
        let b = f64x2::new(3.0, 4.0);
        #[rustfmt::skip]
        let c = f64x2::new(5.0, 6.0);
        #[rustfmt::skip]
        let r = f64x2::new(16.0, 26.0);

        assert_eq!(
            r,
            mem::transmute(__msa_fmadd_d(
                mem::transmute(a),
                mem::transmute(b),
                mem::transmute(c)
            ))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fmax_w() {
        #[rustfmt::skip]
        let a = f32x4::new(1.0, -6.0, 7.0, 8.0);
        #[rustfmt::skip]
        let b = f32x4::new(5.0, -2.0, 3.0, 4.0);
        #[rustfmt::skip]
        let r = f32x4::new(5.0, -2.0, 7.0, 8.0);

        assert_eq!(
            r,
            mem::transmute(__msa_fmax_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fmax_d() {
        #[rustfmt::skip]
        let a = f64x2::new(1.0, 4.0);
        #[rustfmt::skip]
        let b = f64x2::new(3.0, 2.0);
        #[rustfmt::skip]
        let r = f64x2::new(3.0, 4.0);

        assert_eq!(
            r,
            mem::transmute(__msa_fmax_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fmax_a_w() {
        #[rustfmt::skip]
        let a = f32x4::new(1.0, -6.0, -7.0, -8.0);
        #[rustfmt::skip]
        let b = f32x4::new(5.0, -2.0, 3.0, 4.0);
        #[rustfmt::skip]
        let r = f32x4::new(5.0, -6.0, -7.0, -8.0);

        assert_eq!(
            r,
            mem::transmute(__msa_fmax_a_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fmax_a_d() {
        #[rustfmt::skip]
        let a = f64x2::new(1.0, -4.0);
        #[rustfmt::skip]
        let b = f64x2::new(3.0, 2.0);
        #[rustfmt::skip]
        let r = f64x2::new(3.0, -4.0);

        assert_eq!(
            r,
            mem::transmute(__msa_fmax_a_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fmin_w() {
        #[rustfmt::skip]
        let a = f32x4::new(1.0, -6.0, 7.0, 8.0);
        #[rustfmt::skip]
        let b = f32x4::new(5.0, -2.0, 3.0, 4.0);
        #[rustfmt::skip]
        let r = f32x4::new(1.0, -6.0, 3.0, 4.0);

        assert_eq!(
            r,
            mem::transmute(__msa_fmin_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fmin_d() {
        #[rustfmt::skip]
        let a = f64x2::new(1.0, 4.0);
        #[rustfmt::skip]
        let b = f64x2::new(3.0, 2.0);
        #[rustfmt::skip]
        let r = f64x2::new(1.0, 2.0);

        assert_eq!(
            r,
            mem::transmute(__msa_fmin_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fmin_a_w() {
        #[rustfmt::skip]
        let a = f32x4::new(1.0, -6.0, -7.0, -8.0);
        #[rustfmt::skip]
        let b = f32x4::new(5.0, -2.0, 3.0, 4.0);
        #[rustfmt::skip]
        let r = f32x4::new(1.0, -2.0, 3.0, 4.0);

        assert_eq!(
            r,
            mem::transmute(__msa_fmin_a_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fmin_a_d() {
        #[rustfmt::skip]
        let a = f64x2::new(1.0, -4.0);
        #[rustfmt::skip]
        let b = f64x2::new(3.0, 2.0);
        #[rustfmt::skip]
        let r = f64x2::new(1.0, 2.0);

        assert_eq!(
            r,
            mem::transmute(__msa_fmin_a_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fmsub_w() {
        #[rustfmt::skip]
        let a = f32x4::new(1.0, 2.0, 3.0, 4.0);
        #[rustfmt::skip]
        let b = f32x4::new(5.0, 6.0, 7.0, 8.0);
        #[rustfmt::skip]
        let c = f32x4::new(9.0, 10.0, 11.0, 12.0);
        #[rustfmt::skip]
        let r = f32x4::new(-44.0, -58.0, -74.0, -92.0);

        assert_eq!(
            r,
            mem::transmute(__msa_fmsub_w(
                mem::transmute(a),
                mem::transmute(b),
                mem::transmute(c)
            ))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fmsub_d() {
        #[rustfmt::skip]
        let a = f64x2::new(1.0, 2.0);
        #[rustfmt::skip]
        let b = f64x2::new(3.0, 4.0);
        #[rustfmt::skip]
        let c = f64x2::new(5.0, 6.0);
        #[rustfmt::skip]
        let r = f64x2::new(-14.0, -22.0);

        assert_eq!(
            r,
            mem::transmute(__msa_fmsub_d(
                mem::transmute(a),
                mem::transmute(b),
                mem::transmute(c)
            ))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fmul_w() {
        #[rustfmt::skip]
        let a = f32x4::new(1.1, -2.2, 3.3, 4.4);
        #[rustfmt::skip]
        let b = f32x4::new(4.4, 3.3, 2.2, -1.1);
        #[rustfmt::skip]
        let r = f32x4::new(4.84, -7.26, 7.26, -4.84);

        assert_eq!(
            r,
            mem::transmute(__msa_fmul_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fmul_d() {
        #[rustfmt::skip]
        let a = f64x2::new(1.1, -2.2);
        #[rustfmt::skip]
        let b = f64x2::new(4.0, -3.3);
        #[rustfmt::skip]
        let r = f64x2::new(4.4, 7.26);

        assert_eq!(
            r,
            mem::transmute(__msa_fmul_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_frint_w() {
        #[rustfmt::skip]
        let a = f32x4::new(2.6, -2.7, 1.3, -1.7);
        #[rustfmt::skip]
        let r = f32x4::new(3.0, -3.0, 1.0, -2.0);

        assert_eq!(r, mem::transmute(__msa_frint_w(mem::transmute(a))));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_frint_d() {
        #[rustfmt::skip]
        let a = f64x2::new(2.6, 1.3);
        #[rustfmt::skip]
        let r = f64x2::new(3.0, 1.0);

        assert_eq!(r, mem::transmute(__msa_frint_d(mem::transmute(a))));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_frcp_w() {
        #[rustfmt::skip]
        let a = f32x4::new(2.6, -2.7, 1.3, -1.7);
        #[rustfmt::skip]
        let r = f32x4::new(
            0.3846154, -0.37037036, 
            0.7692308, -0.58823526
        );

        assert_eq!(r, mem::transmute(__msa_frcp_w(mem::transmute(a))));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_frcp_d() {
        #[rustfmt::skip]
        let a = f64x2::new(2.6, 1.3);
        #[rustfmt::skip]
        let r = f64x2::new(0.3846153846153846, 0.7692307692307692);

        assert_eq!(r, mem::transmute(__msa_frcp_d(mem::transmute(a))));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_frsqrt_w() {
        #[rustfmt::skip]
        let a = f32x4::new(2.6, 2.7, 1.3, 1.7);
        #[rustfmt::skip]
        let r = f32x4::new(
            0.6201737, 0.6085806, 
            0.87705797, 0.766965
        );

        assert_eq!(r, mem::transmute(__msa_frsqrt_w(mem::transmute(a))));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_frsqrt_d() {
        #[rustfmt::skip]
        let a = f64x2::new(2.6, 1.3);
        #[rustfmt::skip]
        let r = f64x2::new(0.6201736729460422, 0.8770580193070292);

        assert_eq!(r, mem::transmute(__msa_frsqrt_d(mem::transmute(a))));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fsaf_w() {
        #[rustfmt::skip]
        let a = f32x4::new(-5.5, 5.5, 5.5, 5.5);
        #[rustfmt::skip]
        let b = f32x4::new(-5.5, 5.5, 5.5, 5.5);
        #[rustfmt::skip]
        let r = i32x4::new(0, 0, 0, 0);

        assert_eq!(
            r,
            mem::transmute(__msa_fsaf_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fsaf_d() {
        #[rustfmt::skip]
        let a = f64x2::new(-125.5, 5.5);
        #[rustfmt::skip]
        let b = f64x2::new(125.5, 3.3);
        #[rustfmt::skip]
        let r = i64x2::new(0, 0);

        assert_eq!(
            r,
            mem::transmute(__msa_fsaf_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fseq_w() {
        #[rustfmt::skip]
        let a = f32x4::new(-5.5, -3.3, f32::NAN, f32::NAN);
        #[rustfmt::skip]
        let b = f32x4::new(5.5, -3.3, f32::NAN, 1.1);
        #[rustfmt::skip]
        let r = i32x4::new(0, -1, 0, 0);

        assert_eq!(
            r,
            mem::transmute(__msa_fseq_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fseq_d() {
        #[rustfmt::skip]
        let a = f64x2::new(-125.5, 5.5);
        #[rustfmt::skip]
        let b = f64x2::new(125.5, 5.5);
        #[rustfmt::skip]
        let r = i64x2::new(0, -1);

        assert_eq!(
            r,
            mem::transmute(__msa_fseq_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fsle_w() {
        #[rustfmt::skip]
        let a = f32x4::new(5.5, 5.5, 5.5, f32::NAN);
        #[rustfmt::skip]
        let b = f32x4::new(-5.5, 3.3, 5.5, f32::NAN);
        #[rustfmt::skip]
        let r = i32x4::new(0, 0, -1, 0);

        assert_eq!(
            r,
            mem::transmute(__msa_fsle_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fsle_d() {
        #[rustfmt::skip]
        let a = f64x2::new(-125.5, 5.5);
        #[rustfmt::skip]
        let b = f64x2::new(125.5, 3.3);
        #[rustfmt::skip]
        let r = i64x2::new(-1, 0);

        assert_eq!(
            r,
            mem::transmute(__msa_fsle_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fslt_w() {
        #[rustfmt::skip]
        let a = f32x4::new(-5.5, 5.5, 5.5, 5.5);
        #[rustfmt::skip]
        let b = f32x4::new(5.5, 3.3, 5.5, 1.1);
        #[rustfmt::skip]
        let r = i32x4::new(-1, 0, 0, 0);

        assert_eq!(
            r,
            mem::transmute(__msa_fslt_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fslt_d() {
        #[rustfmt::skip]
        let a = f64x2::new(-125.5, 5.5);
        #[rustfmt::skip]
        let b = f64x2::new(125.5, 3.3);
        #[rustfmt::skip]
        let r = i64x2::new(-1, 0);

        assert_eq!(
            r,
            mem::transmute(__msa_fslt_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fsne_w() {
        #[rustfmt::skip]
        let a = f32x4::new(-5.5, 5.5, 5.5, 5.5);
        #[rustfmt::skip]
        let b = f32x4::new(5.5, 3.3, 5.5, 1.1);
        #[rustfmt::skip]
        let r = i32x4::new(-1, -1, 0, -1);

        assert_eq!(
            r,
            mem::transmute(__msa_fsne_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fsne_d() {
        #[rustfmt::skip]
        let a = f64x2::new(-125.5, 5.5);
        #[rustfmt::skip]
        let b = f64x2::new(125.5, 5.5);
        #[rustfmt::skip]
        let r = i64x2::new(-1, 0);

        assert_eq!(
            r,
            mem::transmute(__msa_fsne_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fsor_w() {
        #[rustfmt::skip]
        let a = f32x4::new(-5.5, f32::NAN, 5.5, 5.5);
        #[rustfmt::skip]
        let b = f32x4::new(5.5, 3.3, 5.5, 1.1);
        #[rustfmt::skip]
        let r = i32x4::new(-1, 0, -1, -1);

        assert_eq!(
            r,
            mem::transmute(__msa_fsor_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fsor_d() {
        #[rustfmt::skip]
        let a = f64x2::new(-125.5, 5.5);
        #[rustfmt::skip]
        let b = f64x2::new(125.5, f64::NAN);
        #[rustfmt::skip]
        let r = i64x2::new(-1, 0);

        assert_eq!(
            r,
            mem::transmute(__msa_fsor_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fsqrt_w() {
        #[rustfmt::skip]
        let a = f32x4::new(9.0, 81.0, 1089.0, 10000.0);
        #[rustfmt::skip]
        let r = f32x4::new(3.0, 9.0, 33.0, 100.0);

        assert_eq!(r, mem::transmute(__msa_fsqrt_w(mem::transmute(a))));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fsqrt_d() {
        #[rustfmt::skip]
        let a = f64x2::new(81.0, 10000.0);
        #[rustfmt::skip]
        let r = f64x2::new(9.0, 100.0);

        assert_eq!(r, mem::transmute(__msa_fsqrt_d(mem::transmute(a))));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fsub_w() {
        #[rustfmt::skip]
        let a = f32x4::new(5.5, 6.5, 7.5, 8.5);
        #[rustfmt::skip]
        let b = f32x4::new(1.25, 1.75, 2.25, 2.75);
        #[rustfmt::skip]
        let r = f32x4::new(4.25, 4.75, 5.25, 5.75);

        assert_eq!(
            r,
            mem::transmute(__msa_fsub_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fsub_d() {
        #[rustfmt::skip]
        let a = f64x2::new(555.5, 55.5);
        #[rustfmt::skip]
        let b = f64x2::new(4.25, 3.25);
        #[rustfmt::skip]
        let r = f64x2::new(551.25, 52.25);

        assert_eq!(
            r,
            mem::transmute(__msa_fsub_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fsueq_w() {
        #[rustfmt::skip]
        let a = f32x4::new(5.5, f32::NAN, 5.5, 5.5);
        #[rustfmt::skip]
        let b = f32x4::new(5.5, 5.5, -5.5, 5.5);
        #[rustfmt::skip]
        let r = i32x4::new(-1, -1, 0, -1);

        assert_eq!(
            r,
            mem::transmute(__msa_fsueq_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fsueq_d() {
        #[rustfmt::skip]
        let a = f64x2::new(-5.5, 5.5);
        #[rustfmt::skip]
        let b = f64x2::new(5.5, f64::NAN);
        #[rustfmt::skip]
        let r = i64x2::new(0, -1);

        assert_eq!(
            r,
            mem::transmute(__msa_fsueq_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fsule_w() {
        #[rustfmt::skip]
        let a = f32x4::new(5.7, 5.8, 5.9, f32::NAN);
        #[rustfmt::skip]
        let b = f32x4::new(5.6, 5.9, 5.9, f32::NAN);
        #[rustfmt::skip]
        let r = i32x4::new(0, -1, -1, -1);

        assert_eq!(
            r,
            mem::transmute(__msa_fsule_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fsule_d() {
        #[rustfmt::skip]
        let a = f64x2::new(5.5, 5.5);
        #[rustfmt::skip]
        let b = f64x2::new(5.5, 5.5);
        #[rustfmt::skip]
        let r = i64x2::new(-1, -1);

        assert_eq!(
            r,
            mem::transmute(__msa_fsule_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fsult_w() {
        #[rustfmt::skip]
        let a = f32x4::new(5.5, 5.5, 5.5, 5.5);
        #[rustfmt::skip]
        let b = f32x4::new(5.6, f32::NAN, 2.2, 1.1);
        #[rustfmt::skip]
        let r = i32x4::new(-1, -1, 0, 0);

        assert_eq!(
            r,
            mem::transmute(__msa_fsult_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fsult_d() {
        #[rustfmt::skip]
        let a = f64x2::new(5.5, f64::NAN);
        #[rustfmt::skip]
        let b = f64x2::new(4.4, 3.3);
        #[rustfmt::skip]
        let r = i64x2::new(0, -1);

        assert_eq!(
            r,
            mem::transmute(__msa_fsult_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fsun_w() {
        #[rustfmt::skip]
        let a = f32x4::new(5.5, 5.5, f32::NAN, 5.5);
        #[rustfmt::skip]
        let b = f32x4::new(4.4, 3.3, 2.2, f32::NAN);
        #[rustfmt::skip]
        let r = i32x4::new(0, 0, -1, -1);

        assert_eq!(
            r,
            mem::transmute(__msa_fsun_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fsun_d() {
        #[rustfmt::skip]
        let a = f64x2::new(5.5, f64::NAN);
        #[rustfmt::skip]
        let b = f64x2::new(4.4, 3.3);
        #[rustfmt::skip]
        let r = i64x2::new(0, -1);

        assert_eq!(
            r,
            mem::transmute(__msa_fsun_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fsune_w() {
        #[rustfmt::skip]
        let a = f32x4::new(5.5, 5.5, f32::NAN, 5.5);
        #[rustfmt::skip]
        let b = f32x4::new(4.4, 3.3, 2.2, 5.5);
        #[rustfmt::skip]
        let r = i32x4::new(-1, -1, -1, 0);

        assert_eq!(
            r,
            mem::transmute(__msa_fsune_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_fsune_d() {
        #[rustfmt::skip]
        let a = f64x2::new(5.5, f64::NAN);
        #[rustfmt::skip]
        let b = f64x2::new(5.5, 3.3);
        #[rustfmt::skip]
        let r = i64x2::new(0, -1);

        assert_eq!(
            r,
            mem::transmute(__msa_fsune_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ftint_s_w() {
        #[rustfmt::skip]
        let a = f32x4::new(-5.5, 75.6, -1000.7, 1219.3);
        #[rustfmt::skip]
        let r = i32x4::new(-6, 76, -1001, 1219);

        assert_eq!(r, mem::transmute(__msa_ftint_s_w(mem::transmute(a))));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ftint_s_d() {
        #[rustfmt::skip]
        let a = f64x2::new(-5.5, 25656.4);
        #[rustfmt::skip]
        let r = i64x2::new(-6, 25656);

        assert_eq!(r, mem::transmute(__msa_ftint_s_d(mem::transmute(a))));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ftint_u_w() {
        #[rustfmt::skip]
        let a = f32x4::new(-5.5, 75.6, -1000.7, 1219.3);
        #[rustfmt::skip]
        let r = u32x4::new(0, 76, 0, 1219);

        assert_eq!(r, mem::transmute(__msa_ftint_u_w(mem::transmute(a))));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ftint_u_d() {
        #[rustfmt::skip]
        let a = f64x2::new(5.5, -25656.4);
        #[rustfmt::skip]
        let r = u64x2::new(6, 0);

        assert_eq!(r, mem::transmute(__msa_ftint_u_d(mem::transmute(a))));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ftq_h() {
        #[rustfmt::skip]
        let a = f32x4::new(0.00001, 0.0002, 0.00001, -0.0002);
        #[rustfmt::skip]
        let b = f32x4::new(0.0001, -0.002, 0.0001, 0.002);
        #[rustfmt::skip]
        let r = i16x8::new(3, -66, 3, 66, 0, 7, 0, -7);

        assert_eq!(
            r,
            mem::transmute(__msa_ftq_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ftq_w() {
        #[rustfmt::skip]
        let a = f64x2::new(0.00001, -0.0002);
        #[rustfmt::skip]
        let b = f64x2::new(0.00000045, 0.000015);
        #[rustfmt::skip]
        let r = i32x4::new(966, 32212, 21475, -429497);

        assert_eq!(
            r,
            mem::transmute(__msa_ftq_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ftrunc_s_w() {
        #[rustfmt::skip]
        let a = f32x4::new(-5.5, 75.6, -1000.7, 1219.3);
        #[rustfmt::skip]
        let r = i32x4::new(-5, 75, -1000, 1219);

        assert_eq!(r, mem::transmute(__msa_ftrunc_s_w(mem::transmute(a))));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ftrunc_s_d() {
        #[rustfmt::skip]
        let a = f64x2::new(-5.5, 25656.4);
        #[rustfmt::skip]
        let r = i64x2::new(-5, 25656);

        assert_eq!(r, mem::transmute(__msa_ftrunc_s_d(mem::transmute(a))));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ftrunc_u_w() {
        #[rustfmt::skip]
        let a = f32x4::new(-5.5, 75.6, -1000.7, 1219.3);
        #[rustfmt::skip]
        let r = u32x4::new(0, 75, 0, 1219);

        assert_eq!(r, mem::transmute(__msa_ftrunc_u_w(mem::transmute(a))));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ftrunc_u_d() {
        #[rustfmt::skip]
        let a = f64x2::new(5.5, -25656.4);
        #[rustfmt::skip]
        let r = u64x2::new(5, 0);

        assert_eq!(r, mem::transmute(__msa_ftrunc_u_d(mem::transmute(a))));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_hadd_s_h() {
        #[rustfmt::skip]
        let a = i8x16::new(
            1, 2, 3, 4,
            -1, -2, -3, -4,
            1, 2, 3, 4,
            -1, -2, -3, -4
        );
        #[rustfmt::skip]
        let b = i8x16::new(
            4, 3, 2, 1,
            4, 3, 2, 1,
            4, 3, 2, 1,
            4, 3, 2, 1
        );
        #[rustfmt::skip]
        let r = i16x8::new(6, 6, 2, -2, 6, 6, 2, -2);

        assert_eq!(
            r,
            mem::transmute(__msa_hadd_s_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_hadd_s_w() {
        #[rustfmt::skip]
        let a = i16x8::new(
            1, 2, 3, 4,
            -1, -2, -3, -4
        );
        #[rustfmt::skip]
        let b = i16x8::new(
            4, 3, 2, 1,
            4, 3, 2, 1
        );
        #[rustfmt::skip]
        let r = i32x4::new(6, 6, 2, -2);

        assert_eq!(
            r,
            mem::transmute(__msa_hadd_s_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_hadd_s_d() {
        #[rustfmt::skip]
        let a = i32x4::new(1, -2, 3, -4);
        #[rustfmt::skip]
        let b = i32x4::new(4, 3, 2, 1);
        #[rustfmt::skip]
        let r = i64x2::new(2, -2);

        assert_eq!(
            r,
            mem::transmute(__msa_hadd_s_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_hadd_u_h() {
        #[rustfmt::skip]
        let a = u8x16::new(
            1, 2, 3, 4,
            1, 2, 3, 4,
            1, 2, 3, 4,
            1, 2, 3, 4
        );
        #[rustfmt::skip]
        let b = u8x16::new(
            4, 3, 2, 1,
            4, 3, 2, 1,
            4, 3, 2, 1,
            4, 3, 2, 1
        );
        #[rustfmt::skip]
        let r = u16x8::new(6, 6, 6, 6, 6, 6, 6, 6);

        assert_eq!(
            r,
            mem::transmute(__msa_hadd_u_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_hadd_u_w() {
        #[rustfmt::skip]
        let a = u16x8::new(
            1, 2, 3, 4,
            1, 2, 3, 4
        );
        #[rustfmt::skip]
        let b = u16x8::new(
            4, 3, 2, 1,
            4, 3, 2, 1
        );
        #[rustfmt::skip]
        let r = u32x4::new(6, 6, 6, 6);

        assert_eq!(
            r,
            mem::transmute(__msa_hadd_u_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_hadd_u_d() {
        #[rustfmt::skip]
        let a = u32x4::new(1, 2, 3, 4);
        #[rustfmt::skip]
        let b = u32x4::new(4, 3, 2, 1);
        #[rustfmt::skip]
        let r = u64x2::new(6, 6);

        assert_eq!(
            r,
            mem::transmute(__msa_hadd_u_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_hsub_s_h() {
        #[rustfmt::skip]
        let a = i8x16::new(
            1, 2, 3, 4,
            -1, -2, -3, -4,
            1, 2, 3, 4,
            -1, -2, -3, -4
        );
        #[rustfmt::skip]
        let b = i8x16::new(
            4, 3, 2, 1,
            4, 3, 2, 1,
            4, 3, 2, 1,
            4, 3, 2, 1
        );
        #[rustfmt::skip]
        let r = i16x8::new(-2, 2, -6, -6, -2, 2, -6, -6);

        assert_eq!(
            r,
            mem::transmute(__msa_hsub_s_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_hsub_s_w() {
        #[rustfmt::skip]
        let a = i16x8::new(
            1, 2, 3, 4,
            -1, -2, -3, -4
        );
        #[rustfmt::skip]
        let b = i16x8::new(
            4, 3, 2, 1,
            4, 3, 2, 1
        );
        #[rustfmt::skip]
        let r = i32x4::new(-2, 2, -6, -6);

        assert_eq!(
            r,
            mem::transmute(__msa_hsub_s_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_hsub_s_d() {
        #[rustfmt::skip]
        let a = i32x4::new(1, -2, 3, -4);
        #[rustfmt::skip]
        let b = i32x4::new(4, 3, 2, 1);
        #[rustfmt::skip]
        let r = i64x2::new(-6, -6);

        assert_eq!(
            r,
            mem::transmute(__msa_hsub_s_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_hsub_u_h() {
        #[rustfmt::skip]
        let a = u8x16::new(
            1, 2, 3, 4,
            1, 2, 3, 4,
            1, 2, 3, 4,
            1, 2, 3, 4
        );
        #[rustfmt::skip]
        let b = u8x16::new(
            4, 3, 2, 1,
            4, 3, 2, 1,
            4, 3, 2, 1,
            4, 3, 2, 1
        );
        #[rustfmt::skip]
        let r = i16x8::new(-2, 2, -2, 2, -2, 2, -2, 2);

        assert_eq!(
            r,
            mem::transmute(__msa_hsub_u_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_hsub_u_w() {
        #[rustfmt::skip]
        let a = u16x8::new(
            1, 2, 3, 4,
            1, 2, 3, 4
        );
        #[rustfmt::skip]
        let b = u16x8::new(
            4, 3, 2, 1,
            4, 3, 2, 1
        );
        #[rustfmt::skip]
        let r = i32x4::new(-2, 2, -2, 2);

        assert_eq!(
            r,
            mem::transmute(__msa_hsub_u_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_hsub_u_d() {
        #[rustfmt::skip]
        let a = u32x4::new(1, 2, 3, 4);
        #[rustfmt::skip]
        let b = u32x4::new(4, 3, 2, 1);
        #[rustfmt::skip]
        let r = i64x2::new(-2, 2);

        assert_eq!(
            r,
            mem::transmute(__msa_hsub_u_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ilvev_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            1, 2, 3, 4,
            1, 2, 3, 4,
            1, 2, 3, 4,
            1, 2, 3, 4
        );
        #[rustfmt::skip]
        let b = i8x16::new(
            4, 3, 2, 1,
            4, 3, 2, 1,
            4, 3, 2, 1,
            4, 3, 2, 1
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            4, 1, 2, 3,
            4, 1, 2, 3,
            4, 1, 2, 3,
            4, 1, 2, 3
        );

        assert_eq!(
            r,
            mem::transmute(__msa_ilvev_b(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ilvev_h() {
        #[rustfmt::skip]
        let a = i16x8::new(
            1, 2, 3, 4,
            1, 2, 3, 4
        );
        #[rustfmt::skip]
        let b = i16x8::new(
            4, 3, 2, 1,
            4, 3, 2, 1
        );
        #[rustfmt::skip]
        let r = i16x8::new(4, 1, 2, 3, 4, 1, 2, 3);

        assert_eq!(
            r,
            mem::transmute(__msa_ilvev_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ilvev_w() {
        #[rustfmt::skip]
        let a = i32x4::new(1, 2, 3, 4);
        #[rustfmt::skip]
        let b = i32x4::new(4, 3, 2, 1);
        #[rustfmt::skip]
        let r = i32x4::new(4, 1, 2, 3);

        assert_eq!(
            r,
            mem::transmute(__msa_ilvev_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ilvev_d() {
        #[rustfmt::skip]
        let a = i64x2::new(1, 2);
        #[rustfmt::skip]
        let b = i64x2::new(4, 3);
        #[rustfmt::skip]
        let r = i64x2::new(4, 1);

        assert_eq!(
            r,
            mem::transmute(__msa_ilvev_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ilvl_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16
        );
        #[rustfmt::skip]
        let b = i8x16::new(
            16, 15, 14, 13,
            12, 11, 10, 9,
            8, 7, 6, 5,
            4, 3, 2, 1
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            8, 9, 7, 10,
            6, 11, 5, 12,
            4, 13, 3, 14,
            2, 15, 1, 16
        );

        assert_eq!(
            r,
            mem::transmute(__msa_ilvl_b(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ilvl_h() {
        #[rustfmt::skip]
        let a = i16x8::new(
            1, 2, 3, 4,
            5, 6, 7, 8
        );
        #[rustfmt::skip]
        let b = i16x8::new(
            8, 7, 6, 5,
            4, 3, 2, 1
        );
        #[rustfmt::skip]
        let r = i16x8::new(4, 5, 3, 6, 2, 7, 1, 8);

        assert_eq!(
            r,
            mem::transmute(__msa_ilvl_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ilvl_w() {
        #[rustfmt::skip]
        let a = i32x4::new(1, 2, 3, 4);
        #[rustfmt::skip]
        let b = i32x4::new(4, 3, 2, 1);
        #[rustfmt::skip]
        let r = i32x4::new(2, 3, 1, 4);

        assert_eq!(
            r,
            mem::transmute(__msa_ilvl_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ilvl_d() {
        #[rustfmt::skip]
        let a = i64x2::new(1, 2);
        #[rustfmt::skip]
        let b = i64x2::new(2, 1);
        #[rustfmt::skip]
        let r = i64x2::new(1, 2);

        assert_eq!(
            r,
            mem::transmute(__msa_ilvl_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ilvod_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16
        );
        #[rustfmt::skip]
        let b = i8x16::new(
            16, 15, 14, 13,
            12, 11, 10, 9,
            8, 7, 6, 5,
            4, 3, 2, 1
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            15, 2, 13, 4,
            11, 6, 9, 8,
            7, 10, 5, 12,
            3, 14, 1, 16
        );

        assert_eq!(
            r,
            mem::transmute(__msa_ilvod_b(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ilvod_h() {
        #[rustfmt::skip]
        let a = i16x8::new(
            1, 2, 3, 4,
            5, 6, 7, 8
        );
        #[rustfmt::skip]
        let b = i16x8::new(
            8, 7, 6, 5,
            4, 3, 2, 1
        );
        #[rustfmt::skip]
        let r = i16x8::new(7, 2, 5, 4, 3, 6, 1, 8);

        assert_eq!(
            r,
            mem::transmute(__msa_ilvod_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ilvod_w() {
        #[rustfmt::skip]
        let a = i32x4::new(1, 2, 3, 4);
        #[rustfmt::skip]
        let b = i32x4::new(4, 3, 2, 1);
        #[rustfmt::skip]
        let r = i32x4::new(3, 2, 1, 4);

        assert_eq!(
            r,
            mem::transmute(__msa_ilvod_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ilvod_d() {
        #[rustfmt::skip]
        let a = i64x2::new(1, 2);
        #[rustfmt::skip]
        let b = i64x2::new(2, 1);
        #[rustfmt::skip]
        let r = i64x2::new(1, 2);

        assert_eq!(
            r,
            mem::transmute(__msa_ilvod_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ilvr_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16
        );
        #[rustfmt::skip]
        let b = i8x16::new(
            16, 15, 14, 13,
            12, 11, 10, 9,
            8, 7, 6, 5,
            4, 3, 2, 1
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            16, 1, 15, 2,
            14, 3, 13, 4,
            12, 5, 11, 6,
            10, 7, 9, 8
        );

        assert_eq!(
            r,
            mem::transmute(__msa_ilvr_b(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ilvr_h() {
        #[rustfmt::skip]
        let a = i16x8::new(
            1, 2, 3, 4,
            5, 6, 7, 8,
        );
        #[rustfmt::skip]
        let b = i16x8::new(
            8, 7, 6, 5,
            4, 3, 2, 1,
        );
        #[rustfmt::skip]
        let r = i16x8::new(8, 1, 7, 2, 6, 3, 5, 4);

        assert_eq!(
            r,
            mem::transmute(__msa_ilvr_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ilvr_w() {
        #[rustfmt::skip]
        let a = i32x4::new(1, 2, 3, 4);
        #[rustfmt::skip]
        let b = i32x4::new(4, 3, 2, 1);
        #[rustfmt::skip]
        let r = i32x4::new(4, 1, 3, 2);

        assert_eq!(
            r,
            mem::transmute(__msa_ilvr_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ilvr_d() {
        #[rustfmt::skip]
        let a = i64x2::new(1, 2);
        #[rustfmt::skip]
        let b = i64x2::new(2, 1);
        #[rustfmt::skip]
        let r = i64x2::new(2, 1);

        assert_eq!(
            r,
            mem::transmute(__msa_ilvr_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_insert_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            -100, 127, 4, 127,
            -100, 127, 4, 127,
            -100, 127, 4, 127,
            -100, 127, 4, 127
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            -100, 127, 4, 127,
            -100, 127, 4, 127,
            -100, 127, 4, 127,
            5, 127, 4, 127
        );

        assert_eq!(r, mem::transmute(__msa_insert_b(mem::transmute(a), 12, 5)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_insert_h() {
        #[rustfmt::skip]
        let a = i16x8::new(
            32767, 3276, 100, 11,
            32767, 3276, 100, 11
        );
        #[rustfmt::skip]
        let r = i16x8::new(
            32767, 3276, 100, 11,
            5, 3276, 100, 11
        );

        assert_eq!(r, mem::transmute(__msa_insert_h(mem::transmute(a), 4, 5)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_insert_w() {
        #[rustfmt::skip]
        let a = i32x4::new(100, 2147483647, 5, -2147483647);
        #[rustfmt::skip]
        let r = i32x4::new(100, 7, 5, -2147483647);

        assert_eq!(r, mem::transmute(__msa_insert_w(mem::transmute(a), 1, 7)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_insert_d() {
        #[rustfmt::skip]
        let a = i64x2::new(3, i64::MAX);
        #[rustfmt::skip]
        let r = i64x2::new(3, 100);

        assert_eq!(r, mem::transmute(__msa_insert_d(mem::transmute(a), 1, 100)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_insve_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            -100, i8::MAX, 4, i8::MAX,
            -100, i8::MAX, 4, i8::MAX,
            -100, i8::MAX, 4, i8::MAX,
            -100, i8::MAX, 4, i8::MAX
        );
        #[rustfmt::skip]
        let b = i8x16::new(
            5, 2, 3, 4,
            1, 2, 3, 4,
            1, 2, 3, 4,
            1, 2, 3, 4
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            -100, 127, 4, 127,
            -100, 127, 4, 127,
            -100, 127, 4, 127,
            5, 127, 4, 127
        );

        assert_eq!(
            r,
            mem::transmute(__msa_insve_b(mem::transmute(a), 12, mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_insve_h() {
        #[rustfmt::skip]
        let a = i16x8::new(
            i16::MAX, 3276, 100, 11,
            i16::MAX, 3276, 100, 11
        );
        #[rustfmt::skip]
        let b = i16x8::new(
            1, 2, 3, 4,
            1, 2, 3, 4
        );
        #[rustfmt::skip]
        let r = i16x8::new(
            32767, 3276, 100, 11,
            1, 3276, 100, 11
        );

        assert_eq!(
            r,
            mem::transmute(__msa_insve_h(mem::transmute(a), 4, mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_insve_w() {
        #[rustfmt::skip]
        let a = i32x4::new(100, 2147483647, 5, -2147483647);
        #[rustfmt::skip]
        let b = i32x4::new(1, 2, 3, 4);
        #[rustfmt::skip]
        let r = i32x4::new(100, 2147483647, 5, 1);

        assert_eq!(
            r,
            mem::transmute(__msa_insve_w(mem::transmute(a), 3, mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_insve_d() {
        #[rustfmt::skip]
        let a = i64x2::new(3, i64::MAX);
        #[rustfmt::skip]
        let b = i64x2::new(1, 2);
        #[rustfmt::skip]
        let r = i64x2::new(3, 1);

        assert_eq!(
            r,
            mem::transmute(__msa_insve_d(mem::transmute(a), 1, mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ld_b() {
        #[rustfmt::skip]
        let mut a : [i8; 32] = [
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31
        ];
        let p = &mut a[4] as *mut _ as *mut u8;
        #[rustfmt::skip]
        let r = i8x16::new(
            13, 14, 15, 16, 
            17, 18, 19, 20, 
            21, 22, 23, 24, 
            25, 26, 27, 28
        );

        assert_eq!(r, mem::transmute(__msa_ld_b(p, 9)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ld_h() {
        #[rustfmt::skip]
        let mut a : [i16; 16] = [
            0, 1, 2, 3, 4, 5, 6, 7, 
            8, 9, 10, 11, 12, 13, 14, 15
        ];
        let p = &mut a[4] as *mut _ as *mut u8;
        #[rustfmt::skip]
        let r = i16x8::new(3, 4, 5, 6, 7, 8, 9, 10);

        assert_eq!(r, mem::transmute(__msa_ld_h(p, -2)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ld_w() {
        #[rustfmt::skip]
        let mut a : [i32; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
        let p = &mut a[3] as *mut _ as *mut u8;
        #[rustfmt::skip]
        let r = i32x4::new(2, 3, 4, 5);

        assert_eq!(r, mem::transmute(__msa_ld_w(p, -4)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ld_d() {
        #[rustfmt::skip]
        let mut a : [i64; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
        let p = &mut a[4] as *mut _ as *mut u8;
        #[rustfmt::skip]
        let r = i64x2::new(0, 1);

        assert_eq!(r, mem::transmute(__msa_ld_d(p, -32)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ldi_b() {
        #[rustfmt::skip]
        let r = i8x16::new(
            -20, -20, -20, -20,
            -20, -20, -20, -20,
            -20, -20, -20, -20,
            -20, -20, -20, -20
        );

        assert_eq!(r, mem::transmute(__msa_ldi_b(-20)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ldi_h() {
        #[rustfmt::skip]
        let r = i16x8::new(
            255, 255, 255, 255,
            255, 255, 255, 255
        );

        assert_eq!(r, mem::transmute(__msa_ldi_h(255)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ldi_w() {
        #[rustfmt::skip]
        let r = i32x4::new(-509, -509, -509, -509);

        assert_eq!(r, mem::transmute(__msa_ldi_w(-509)));
    }

    // FIXME: https://reviews.llvm.org/D59884
    // If target type is i64, negative immediate loses the sign
    // Test passes if 4294967185 is used instead -111 in vector `r`
    // #[simd_test(enable = "msa")]
    // unsafe fn test_msa_ldi_d() {
    //     let r = i64x2::new(-111, -111);

    //     assert_eq!(r, mem::transmute(__msa_ldi_d(-111)));
    // }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_madd_q_h() {
        #[rustfmt::skip]
        let a = i16x8::new(
            i16::MAX, 1024, i16::MIN, -1024,
            1, 2, 3, 4
        );
        #[rustfmt::skip]
        let b = i16x8::new(
            1024, 1024, 1024, 1024,
            1024, 1024, 1024, 1024
        );
        #[rustfmt::skip]
        let c = i16x8::new(
            i16::MAX, i16::MAX, 1, -1,
            33, 66, 99, 132
        );
        #[rustfmt::skip]
        let r = i16x8::new(32767, 2047, -32768, -1025, 2, 4, 6, 8);

        assert_eq!(
            r,
            mem::transmute(__msa_madd_q_h(
                mem::transmute(a),
                mem::transmute(b),
                mem::transmute(c)
            ))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_madd_q_w() {
        #[rustfmt::skip]
        let a = i32x4::new(i32::MAX, i32::MIN, 1, 2);
        #[rustfmt::skip]
        let b = i32x4::new(102401, 102401, 102401, 102401);
        #[rustfmt::skip]
        let c = i32x4::new(10240, 20480, 30720, 40960);
        #[rustfmt::skip]
        let r = i32x4::new(2147483647, -2147483648, 2, 3);

        assert_eq!(
            r,
            mem::transmute(__msa_madd_q_w(
                mem::transmute(a),
                mem::transmute(b),
                mem::transmute(c)
            ))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_maddr_q_h() {
        #[rustfmt::skip]
        let a = i16x8::new(
            32767, 1024, -32768, -1024,
            1, 2, 3, 4
        );
        #[rustfmt::skip]
        let b = i16x8::new(
            1024, 1024, 1024, 1024,
            1024, 1024, 1024, 1024
        );
        #[rustfmt::skip]
        let c = i16x8::new(
            32767, 32767, 32767, 32767,
            33, 66, 99, 132
        );
        #[rustfmt::skip]
        let r = i16x8::new(32767, 2048, -31744, 0, 2, 4, 6, 8);

        assert_eq!(
            r,
            mem::transmute(__msa_maddr_q_h(
                mem::transmute(a),
                mem::transmute(b),
                mem::transmute(c)
            ))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_maddr_q_w() {
        #[rustfmt::skip]
        let a = i32x4::new(i32::MAX, i32::MIN, 1, 2);
        #[rustfmt::skip]
        let b = i32x4::new(102401, 102401, 102401, 102401);
        #[rustfmt::skip]
        let c = i32x4::new(10240, 20480, 30720, 40960);
        #[rustfmt::skip]
        let r = i32x4::new(2147483647, -2147483647, 2, 4);

        assert_eq!(
            r,
            mem::transmute(__msa_maddr_q_w(
                mem::transmute(a),
                mem::transmute(b),
                mem::transmute(c)
            ))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_maddv_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            1, 2, 3, 4,
            1, 2, 3, 4,
            1, 2, 3, 4,
            1, 2, 3, 4
        );
        #[rustfmt::skip]
        let b = i8x16::new(
            5, 6, 7, 8,
            5, 6, 7, 8,
            5, 6, 7, 8,
            5, 6, 7, 8
        );
        #[rustfmt::skip]
        let c = i8x16::new(
            9, 10, 11, 12,
            9, 10, 11, 12,
            9, 10, 11, 12,
            9, 10, 11, 12
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            46, 62, 80, 100,
            46, 62, 80, 100,
            46, 62, 80, 100,
            46, 62, 80, 100
        );

        assert_eq!(
            r,
            mem::transmute(__msa_maddv_b(
                mem::transmute(a),
                mem::transmute(b),
                mem::transmute(c)
            ))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_maddv_h() {
        #[rustfmt::skip]
        let a = i16x8::new(1, 2, 3, 4, 1, 2, 3, 4);
        #[rustfmt::skip]
        let b = i16x8::new(5, 6, 7, 8, 5, 6, 7, 8);
        #[rustfmt::skip]
        let c = i16x8::new(9, 10, 11, 12, 9, 10, 11, 12);
        #[rustfmt::skip]
        let r = i16x8::new(46, 62, 80, 100, 46, 62, 80, 100);

        assert_eq!(
            r,
            mem::transmute(__msa_maddv_h(
                mem::transmute(a),
                mem::transmute(b),
                mem::transmute(c)
            ))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_maddv_w() {
        #[rustfmt::skip]
        let a = i32x4::new(1, 2, 1, 2);
        #[rustfmt::skip]
        let b = i32x4::new(3, 4, 3, 4);
        #[rustfmt::skip]
        let c = i32x4::new(5, 6, 5, 6);
        #[rustfmt::skip]
        let r = i32x4::new(16, 26, 16, 26);

        assert_eq!(
            r,
            mem::transmute(__msa_maddv_w(
                mem::transmute(a),
                mem::transmute(b),
                mem::transmute(c)
            ))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_maddv_d() {
        #[rustfmt::skip]
        let a = i64x2::new(1, 2);
        #[rustfmt::skip]
        let b = i64x2::new(3, 4);
        #[rustfmt::skip]
        let c = i64x2::new(5, 6);
        #[rustfmt::skip]
        let r = i64x2::new(16, 26);

        assert_eq!(
            r,
            mem::transmute(__msa_maddv_d(
                mem::transmute(a),
                mem::transmute(b),
                mem::transmute(c)
            ))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_max_a_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            1, 2, 3, 4,
            -1, -2, -3, -4,
            1, 2, 3, 4,
            -1, -2, -3, -4
        );
        #[rustfmt::skip]
        let b = i8x16::new(
            -6, -7, -8, -9,
            6, 7, 8, 9,
            -6, -7, -8, -9,
            6, 7, 8, 9
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            -6, -7, -8, -9,
            6, 7, 8, 9,
            -6, -7, -8, -9,
            6, 7, 8, 9
        );

        assert_eq!(
            r,
            mem::transmute(__msa_max_a_b(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_max_a_h() {
        #[rustfmt::skip]
        let a = i16x8::new(1, -2, 3, -4, 1, -2, 3, -4);
        #[rustfmt::skip]
        let b = i16x8::new(-6, 7, -8, 9, -6, 7, -8, 9);
        #[rustfmt::skip]
        let r = i16x8::new(-6, 7, -8, 9, -6, 7, -8, 9);

        assert_eq!(
            r,
            mem::transmute(__msa_max_a_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_max_a_w() {
        #[rustfmt::skip]
        let a = i32x4::new(1, -2, 3, -4);
        #[rustfmt::skip]
        let b = i32x4::new(6, 7, 8, 9);
        #[rustfmt::skip]
        let r = i32x4::new(6, 7, 8, 9);

        assert_eq!(
            r,
            mem::transmute(__msa_max_a_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_max_a_d() {
        #[rustfmt::skip]
        let a = i64x2::new(-1, 2);
        #[rustfmt::skip]
        let b = i64x2::new(6, -7);
        #[rustfmt::skip]
        let r = i64x2::new(6, -7);

        assert_eq!(
            r,
            mem::transmute(__msa_max_a_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_max_s_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            1, 2, 3, 4,
            -1, -2, -3, -4,
            1, 2, 3, 4,
            -1, -2, -3, -4
        );
        #[rustfmt::skip]
        let b = i8x16::new(
            -6, -7, -8, -9,
            6, 7, 8, 9,
            -6, -7, -8, -9,
            6, 7, 8, 9
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            1, 2, 3, 4,
            6, 7, 8, 9,
            1, 2, 3, 4,
            6, 7, 8, 9
        );

        assert_eq!(
            r,
            mem::transmute(__msa_max_s_b(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_max_s_h() {
        #[rustfmt::skip]
        let a = i16x8::new(1, -2, 3, -4, 1, -2, 3, -4);
        #[rustfmt::skip]
        let b = i16x8::new(-6, 7, -8, 9, -6, 7, -8, 9);
        #[rustfmt::skip]
        let r = i16x8::new(1, 7, 3, 9, 1, 7, 3, 9);

        assert_eq!(
            r,
            mem::transmute(__msa_max_s_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_max_s_w() {
        #[rustfmt::skip]
        let a = i32x4::new(1, -2, 3, -4);
        #[rustfmt::skip]
        let b = i32x4::new(6, 7, 8, 9);
        #[rustfmt::skip]
        let r = i32x4::new(6, 7, 8, 9);

        assert_eq!(
            r,
            mem::transmute(__msa_max_s_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_max_s_d() {
        #[rustfmt::skip]
        let a = i64x2::new(-1, 2);
        #[rustfmt::skip]
        let b = i64x2::new(6, -7);
        #[rustfmt::skip]
        let r = i64x2::new(6, 2);

        assert_eq!(
            r,
            mem::transmute(__msa_max_s_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_max_u_b() {
        #[rustfmt::skip]
        let a = u8x16::new(
            1, 2, 3, 4,
            1, 2, 3, 4,
            1, 2, 3, 4,
            1, 2, 3, 4
        );
        #[rustfmt::skip]
        let b = u8x16::new(
            6, 7, 8, 9,
            6, 7, 8, 9,
            6, 7, 8, 9,
            6, 7, 8, 9
        );
        #[rustfmt::skip]
        let r = u8x16::new(
            6, 7, 8, 9,
            6, 7, 8, 9,
            6, 7, 8, 9,
            6, 7, 8, 9
        );

        assert_eq!(
            r,
            mem::transmute(__msa_max_u_b(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_max_u_h() {
        #[rustfmt::skip]
        let a = u16x8::new(1, 2, 3, 4, 1, 2, 3, 4);
        #[rustfmt::skip]
        let b = u16x8::new(6, 7, 8, 9, 6, 7, 8, 9);
        #[rustfmt::skip]
        let r = u16x8::new(6, 7, 8, 9, 6, 7, 8, 9);

        assert_eq!(
            r,
            mem::transmute(__msa_max_u_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_max_u_w() {
        #[rustfmt::skip]
        let a = u32x4::new(1, 2, 3, 4);
        #[rustfmt::skip]
        let b = u32x4::new(6, 7, 8, 9);
        #[rustfmt::skip]
        let r = u32x4::new(6, 7, 8, 9);

        assert_eq!(
            r,
            mem::transmute(__msa_max_u_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_max_u_d() {
        #[rustfmt::skip]
        let a = u64x2::new(1, 2);
        #[rustfmt::skip]
        let b = u64x2::new(6, 7);
        #[rustfmt::skip]
        let r = u64x2::new(6, 7);

        assert_eq!(
            r,
            mem::transmute(__msa_max_u_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_maxi_s_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            1, -20, -6, 8,
            1, -20, -6, 8,
            1, -20, -6, 8,
            1, -20, -6, 8
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            1, -16, -6, 8,
            1, -16, -6, 8,
            1, -16, -6, 8,
            1, -16, -6, 8
        );

        assert_eq!(r, mem::transmute(__msa_maxi_s_b(mem::transmute(a), -16)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_maxi_s_h() {
        #[rustfmt::skip]
        let a = i16x8::new(1, 3, -60, -8, 1, 3, -6, -8);
        #[rustfmt::skip]
        let r = i16x8::new(15, 15, 15, 15, 15, 15, 15, 15);

        assert_eq!(r, mem::transmute(__msa_maxi_s_h(mem::transmute(a), 15)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_maxi_s_w() {
        #[rustfmt::skip]
        let a = i32x4::new(1, 3, -6, -8);
        #[rustfmt::skip]
        let r = i32x4::new(1, 3, -5, -5);

        assert_eq!(r, mem::transmute(__msa_maxi_s_w(mem::transmute(a), -5)));
    }

    // FIXME: https://reviews.llvm.org/D59884
    // If target type is i64, negative immediate loses the sign
    // Test passes if 4294967293 is used instead -3 in vector `r`
    // #[simd_test(enable = "msa")]
    // unsafe fn test_msa_maxi_s_d() {
    //     #[rustfmt::skip]
    //     let a = i64x2::new(1, -8);
    //     #[rustfmt::skip]
    //     let r = i64x2::new(-3, -3);

    //     assert_eq!(r, mem::transmute(__msa_maxi_s_d(mem::transmute(a), -3)));
    // }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_maxi_u_b() {
        #[rustfmt::skip]
        let a = u8x16::new(
            1, 3, 6, 8,
            1, 3, 6, 8,
            1, 3, 6, 8,
            1, 3, 6, 8
        );
        #[rustfmt::skip]
        let r = u8x16::new(
            5, 5, 6, 8,
            5, 5, 6, 8,
            5, 5, 6, 8,
            5, 5, 6, 8
        );

        assert_eq!(r, mem::transmute(__msa_maxi_u_b(mem::transmute(a), 5)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_maxi_u_h() {
        #[rustfmt::skip]
        let a = u16x8::new(1, 3, 6, 8, 1, 3, 6, 8);
        #[rustfmt::skip]
        let r = u16x8::new(5, 5, 6, 8, 5, 5, 6, 8);

        assert_eq!(r, mem::transmute(__msa_maxi_u_h(mem::transmute(a), 5)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_maxi_u_w() {
        #[rustfmt::skip]
        let a = u32x4::new(1, 3, 6, 8);
        #[rustfmt::skip]
        let r = u32x4::new(5, 5, 6, 8);

        assert_eq!(r, mem::transmute(__msa_maxi_u_w(mem::transmute(a), 5)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_maxi_u_d() {
        #[rustfmt::skip]
        let a = u64x2::new(1, 8);
        #[rustfmt::skip]
        let r = u64x2::new(5, 8);

        assert_eq!(r, mem::transmute(__msa_maxi_u_d(mem::transmute(a), 5)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_min_a_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            1, 2, 3, 4,
            -1, -2, -3, -4,
            1, 2, 3, 4,
            -1, -2, -3, -4
        );
        #[rustfmt::skip]
        let b = i8x16::new(
            -6, -7, -8, -9,
            6, 7, 8, 9,
            -6, -7, -8, -9,
            6, 7, 8, 9
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            1, 2, 3, 4,
            -1, -2, -3, -4,
            1, 2, 3, 4,
            -1, -2, -3, -4
        );

        assert_eq!(
            r,
            mem::transmute(__msa_min_a_b(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_min_a_h() {
        #[rustfmt::skip]
        let a = i16x8::new(1, -2, 3, -4, 1, -2, 3, -4);
        #[rustfmt::skip]
        let b = i16x8::new(-6, 7, -8, 9, -6, 7, -8, 9);
        #[rustfmt::skip]
        let r = i16x8::new(1, -2, 3, -4, 1, -2, 3, -4);

        assert_eq!(
            r,
            mem::transmute(__msa_min_a_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_min_a_w() {
        #[rustfmt::skip]
        let a = i32x4::new(1, -2, 3, -4);
        #[rustfmt::skip]
        let b = i32x4::new(6, 7, 8, 9);
        #[rustfmt::skip]
        let r = i32x4::new(1, -2, 3, -4);

        assert_eq!(
            r,
            mem::transmute(__msa_min_a_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_min_a_d() {
        #[rustfmt::skip]
        let a = i64x2::new(-1, 2);
        #[rustfmt::skip]
        let b = i64x2::new(6, -7);
        #[rustfmt::skip]
        let r = i64x2::new(-1, 2);

        assert_eq!(
            r,
            mem::transmute(__msa_min_a_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_min_s_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            1, 2, 3, 4,
            -1, -2, -3, -4,
            1, 2, 3, 4,
            -1, -2, -3, -4
        );
        #[rustfmt::skip]
        let b = i8x16::new(
            -6, -7, -8, -9,
            6, 7, 8, 9,
            -6, -7, -8, -9,
            6, 7, 8, 9
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            -6, -7, -8, -9,
            -1, -2, -3, -4,
            -6, -7, -8, -9,
            -1, -2, -3, -4
        );

        assert_eq!(
            r,
            mem::transmute(__msa_min_s_b(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_min_s_h() {
        #[rustfmt::skip]
        let a = i16x8::new(1, -2, 3, -4, 1, -2, 3, -4);
        #[rustfmt::skip]
        let b = i16x8::new(-6, 7, -8, 9, -6, 7, -8, 9);
        #[rustfmt::skip]
        let r = i16x8::new(-6, -2, -8, -4, -6, -2, -8, -4);

        assert_eq!(
            r,
            mem::transmute(__msa_min_s_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_min_s_w() {
        #[rustfmt::skip]
        let a = i32x4::new(1, -2, 3, -4);
        #[rustfmt::skip]
        let b = i32x4::new(6, 7, 8, 9);
        #[rustfmt::skip]
        let r = i32x4::new(1, -2, 3, -4);

        assert_eq!(
            r,
            mem::transmute(__msa_min_s_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_min_s_d() {
        #[rustfmt::skip]
        let a = i64x2::new(-1, 2);
        #[rustfmt::skip]
        let b = i64x2::new(6, -7);
        #[rustfmt::skip]
        let r = i64x2::new(-1, -7);

        assert_eq!(
            r,
            mem::transmute(__msa_min_s_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_mini_s_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            1, 2, 3, 4,
            -1, -2, -3, -4,
            1, 2, 3, 4,
            -1, -2, -3, -4
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            -10, -10, -10, -10,
            -10, -10, -10, -10,
            -10, -10, -10, -10,
            -10, -10, -10, -10
        );

        assert_eq!(r, mem::transmute(__msa_mini_s_b(mem::transmute(a), -10)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_mini_s_h() {
        #[rustfmt::skip]
        let a = i16x8::new(1, -2, 3, -4, 1, -2, 3, -4);
        #[rustfmt::skip]
        let r = i16x8::new(-3, -3, -3, -4, -3, -3, -3, -4);

        assert_eq!(r, mem::transmute(__msa_mini_s_h(mem::transmute(a), -3)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_mini_s_w() {
        #[rustfmt::skip]
        let a = i32x4::new(1, -2, 3, -4);
        #[rustfmt::skip]
        let r = i32x4::new(-3, -3, -3, -4);

        assert_eq!(r, mem::transmute(__msa_mini_s_w(mem::transmute(a), -3)));
    }

    // FIXME: https://reviews.llvm.org/D59884
    // If target type is i64, negative immediate loses the sign
    // -3 is represented as 4294967293
    // #[simd_test(enable = "msa")]
    // unsafe fn test_msa_mini_s_d() {
    //     #[rustfmt::skip]
    //     let a = i64x2::new(-3, 2);
    //     #[rustfmt::skip]
    //     let r = i64x2::new(-1, -3);

    //     assert_eq!(r, mem::transmute(__msa_mini_s_d(mem::transmute(a), -3)));
    // }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_min_u_b() {
        #[rustfmt::skip]
        let a = u8x16::new(
            1, 2, 3, 4,
            1, 2, 3, 4,
            1, 2, 3, 4,
            1, 2, 3, 4
        );
        #[rustfmt::skip]
        let b = u8x16::new(
            6, 7, 8, 9,
            6, 7, 8, 9,
            6, 7, 8, 9,
            6, 7, 8, 9
        );
        #[rustfmt::skip]
        let r = u8x16::new(
            1, 2, 3, 4,
            1, 2, 3, 4,
            1, 2, 3, 4,
            1, 2, 3, 4
        );

        assert_eq!(
            r,
            mem::transmute(__msa_min_u_b(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_min_u_h() {
        #[rustfmt::skip]
        let a = u16x8::new(1, 2, 3, 4, 1, 2, 3, 4);
        #[rustfmt::skip]
        let b = u16x8::new(6, 7, 8, 9, 6, 7, 8, 9);
        #[rustfmt::skip]
        let r = u16x8::new(1, 2, 3, 4, 1, 2, 3, 4,);

        assert_eq!(
            r,
            mem::transmute(__msa_min_u_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_min_u_w() {
        #[rustfmt::skip]
        let a = u32x4::new(1, 2, 3, 4);
        #[rustfmt::skip]
        let b = u32x4::new(6, 7, 8, 9);
        #[rustfmt::skip]
        let r = u32x4::new(1, 2, 3, 4,);

        assert_eq!(
            r,
            mem::transmute(__msa_min_u_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_min_u_d() {
        #[rustfmt::skip]
        let a = u64x2::new(1, 2);
        #[rustfmt::skip]
        let b = u64x2::new(6, 7);
        #[rustfmt::skip]
        let r = u64x2::new(1, 2,);

        assert_eq!(
            r,
            mem::transmute(__msa_min_u_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_mini_u_b() {
        #[rustfmt::skip]
        let a = u8x16::new(
            1, 3, 6, 8,
            1, 3, 6, 8,
            1, 3, 6, 8,
            1, 3, 6, 8
        );
        #[rustfmt::skip]
        let r = u8x16::new(
            1, 3, 5, 5,
            1, 3, 5, 5,
            1, 3, 5, 5,
            1, 3, 5, 5
        );

        assert_eq!(r, mem::transmute(__msa_mini_u_b(mem::transmute(a), 5)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_mini_u_h() {
        #[rustfmt::skip]
        let a = u16x8::new(1, 3, 6, 8, 1, 3, 6, 8);
        #[rustfmt::skip]
        let r = u16x8::new(1, 3, 5, 5, 1, 3, 5, 5);

        assert_eq!(r, mem::transmute(__msa_mini_u_h(mem::transmute(a), 5)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_mini_u_w() {
        #[rustfmt::skip]
        let a = u32x4::new(1, 3, 6, 8);
        #[rustfmt::skip]
        let r = u32x4::new(1, 3, 5, 5);

        assert_eq!(r, mem::transmute(__msa_mini_u_w(mem::transmute(a), 5)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_mini_u_d() {
        #[rustfmt::skip]
        let a = u64x2::new(1, 8);
        #[rustfmt::skip]
        let r = u64x2::new(1, 5);

        assert_eq!(r, mem::transmute(__msa_mini_u_d(mem::transmute(a), 5)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_mod_s_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            -6, -7, -8, -9,
            6, 7, 8, 9,
            -6, -7, -8, -9,
            6, 7, 8, 9
        );
        #[rustfmt::skip]
        let b = i8x16::new(
            1, 2, 3, 4,
            -1, -2, -3, -4,
            1, 2, 3, 4,
            -1, -2, -3, -4
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            0, -1, -2, -1,
            0, 1, 2, 1,
            0, -1, -2, -1,
            0, 1, 2, 1
        );

        assert_eq!(
            r,
            mem::transmute(__msa_mod_s_b(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_mod_s_h() {
        #[rustfmt::skip]
        let a = i16x8::new(-6, 7, -8, 9, -6, 7, -8, 9);
        #[rustfmt::skip]
        let b = i16x8::new(1, -2, 3, -4, 1, -2, 3, -4);
        #[rustfmt::skip]
        let r = i16x8::new(0, 1, -2, 1, 0, 1, -2, 1);

        assert_eq!(
            r,
            mem::transmute(__msa_mod_s_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_mod_s_w() {
        #[rustfmt::skip]
        let a = i32x4::new(6, 7, 8, 9);
        #[rustfmt::skip]
        let b = i32x4::new(1, -2, 3, -4);
        #[rustfmt::skip]
        let r = i32x4::new(0, 1, 2, 1);

        assert_eq!(
            r,
            mem::transmute(__msa_mod_s_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_mod_s_d() {
        #[rustfmt::skip]
        let a = i64x2::new(6, -7);
        #[rustfmt::skip]
        let b = i64x2::new(-1, 2);
        #[rustfmt::skip]
        let r = i64x2::new(0, -1);

        assert_eq!(
            r,
            mem::transmute(__msa_mod_s_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_mod_u_b() {
        #[rustfmt::skip]
        let a = u8x16::new(
            6, 7, 8, 9,
            6, 7, 8, 9,
            6, 7, 8, 9,
            6, 7, 8, 9
        );
        #[rustfmt::skip]
        let b = u8x16::new(
            1, 2, 3, 4,
            1, 2, 3, 4,
            1, 2, 3, 4,
            1, 2, 3, 4
        );
        #[rustfmt::skip]
        let r = u8x16::new(
            0, 1, 2, 1,
            0, 1, 2, 1,
            0, 1, 2, 1,
            0, 1, 2, 1
        );

        assert_eq!(
            r,
            mem::transmute(__msa_mod_u_b(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_mod_u_h() {
        #[rustfmt::skip]
        let a = u16x8::new(6, 7, 8, 9, 6, 7, 8, 9);
        #[rustfmt::skip]
        let b = u16x8::new(1, 2, 3, 4, 1, 2, 3, 4);
        #[rustfmt::skip]
        let r = u16x8::new(0, 1, 2, 1, 0, 1, 2, 1);

        assert_eq!(
            r,
            mem::transmute(__msa_mod_u_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_mod_u_w() {
        #[rustfmt::skip]
        let a = u32x4::new(6, 7, 8, 9);
        #[rustfmt::skip]
        let b = u32x4::new(1, 2, 3, 4);
        #[rustfmt::skip]
        let r = u32x4::new(0, 1, 2, 1);

        assert_eq!(
            r,
            mem::transmute(__msa_mod_u_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_mod_u_d() {
        #[rustfmt::skip]
        let a = u64x2::new(6, 7);
        #[rustfmt::skip]
        let b = u64x2::new(1, 2);
        #[rustfmt::skip]
        let r = u64x2::new(0, 1);

        assert_eq!(
            r,
            mem::transmute(__msa_mod_u_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_move_v() {
        #[rustfmt::skip]
        let a = i8x16::new(
            1, 2, 3, 4,
            5, 6, 7, 8,
            1, 2, 3, 4,
            5, 6, 7, 8
            );
        #[rustfmt::skip]
        let r = i8x16::new(
            1, 2, 3, 4,
            5, 6, 7, 8,
            1, 2, 3, 4,
            5, 6, 7, 8
            );

        assert_eq!(r, mem::transmute(__msa_move_v(mem::transmute(a))));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_msub_q_h() {
        #[rustfmt::skip]
        let a = i16x8::new(
            1024, -1024, 1024, -1024,
            1, 2, 3, 4
        );
        #[rustfmt::skip]
        let b = i16x8::new(
            1025, 1025, 1025, 1025,
            1025, 1025, 1025, 1025
        );
        #[rustfmt::skip]
        let c = i16x8::new(
            1024, 2048, 3072, 4096,
            1024, 2048, 3072, 4096
        );
        #[rustfmt::skip]
        let r = i16x8::new(991, -1089, 927, -1153, -32, -63, -94, -125);

        assert_eq!(
            r,
            mem::transmute(__msa_msub_q_h(
                mem::transmute(a),
                mem::transmute(b),
                mem::transmute(c)
            ))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_msub_q_w() {
        #[rustfmt::skip]
        let a = i32x4::new(2147483647, -2147483647, 1, 2);
        #[rustfmt::skip]
        let b = i32x4::new(10240, 10240, 10240, 10240);
        #[rustfmt::skip]
        let c = i32x4::new(10240, 20480, 30720, 40960);
        #[rustfmt::skip]
        let r = i32x4::new(2147483646, -2147483648, 0, 1);

        assert_eq!(
            r,
            mem::transmute(__msa_msub_q_w(
                mem::transmute(a),
                mem::transmute(b),
                mem::transmute(c)
            ))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_msubr_q_h() {
        #[rustfmt::skip]
        let a = i16x8::new(
            1024, -1024, 1024, -1024,
            1, 2, 3, 4
        );
        #[rustfmt::skip]
        let b = i16x8::new(
            1025, 1025, 1025, 1025, 
            1025, 1025, 1025, 1025
        );
        #[rustfmt::skip]
        let c = i16x8::new(
            1024, 2048, 3072, 4096,
            1024, 2048, 3072, 4096
        );
        #[rustfmt::skip]
        let r = i16x8::new(992, -1088, 928, -1152, -31, -62, -93, -124);

        assert_eq!(
            r,
            mem::transmute(__msa_msubr_q_h(
                mem::transmute(a),
                mem::transmute(b),
                mem::transmute(c)
            ))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_msubr_q_w() {
        #[rustfmt::skip]
        let a = i32x4::new(i32::MAX, -2147483647, 1, 2);
        #[rustfmt::skip]
        let b = i32x4::new(10240, 10240, 10240, 10240);
        #[rustfmt::skip]
        let c = i32x4::new(10240, 20480, 30720, 40960);
        #[rustfmt::skip]
        let r = i32x4::new(2147483647, -2147483647, 1, 2);

        assert_eq!(
            r,
            mem::transmute(__msa_msubr_q_w(
                mem::transmute(a),
                mem::transmute(b),
                mem::transmute(c)
            ))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_msubv_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            1, 2, 3, 4,
            1, 2, 3, 4,
            1, 2, 3, 4,
            1, 2, 3, 4
        );
        #[rustfmt::skip]
        let b = i8x16::new(
            5, 6, 7, 8,
            5, 6, 7, 8,
            5, 6, 7, 8,
            5, 6, 7, 8
        );
        #[rustfmt::skip]
        let c = i8x16::new(
            9, 10, 11, 12,
            9, 10, 11, 12,
            9, 10, 11, 12,
            9, 10, 11, 12
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            -44, -58, -74, -92,
            -44, -58, -74, -92,
            -44, -58, -74, -92,
            -44, -58, -74, -92
        );

        assert_eq!(
            r,
            mem::transmute(__msa_msubv_b(
                mem::transmute(a),
                mem::transmute(b),
                mem::transmute(c)
            ))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_msubv_h() {
        #[rustfmt::skip]
        let a = i16x8::new(1, 2, 3, 4, 1, 2, 3, 4);
        #[rustfmt::skip]
        let b = i16x8::new(5, 6, 7, 8, 5, 6, 7, 8);
        #[rustfmt::skip]
        let c = i16x8::new(9, 10, 11, 12, 9, 10, 11, 12);
        #[rustfmt::skip]
        let r = i16x8::new(-44, -58, -74, -92, -44, -58, -74, -92);

        assert_eq!(
            r,
            mem::transmute(__msa_msubv_h(
                mem::transmute(a),
                mem::transmute(b),
                mem::transmute(c)
            ))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_msubv_w() {
        #[rustfmt::skip]
        let a = i32x4::new(1, 2, 1, 2);
        #[rustfmt::skip]
        let b = i32x4::new(3, 4, 3, 4);
        #[rustfmt::skip]
        let c = i32x4::new(5, 6, 5, 6);
        #[rustfmt::skip]
        let r = i32x4::new(-14, -22, -14, -22);

        assert_eq!(
            r,
            mem::transmute(__msa_msubv_w(
                mem::transmute(a),
                mem::transmute(b),
                mem::transmute(c)
            ))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_msubv_d() {
        #[rustfmt::skip]
        let a = i64x2::new(1, 2);
        #[rustfmt::skip]
        let b = i64x2::new(3, 4);
        #[rustfmt::skip]
        let c = i64x2::new(5, 6);
        #[rustfmt::skip]
        let r = i64x2::new(-14, -22);

        assert_eq!(
            r,
            mem::transmute(__msa_msubv_d(
                mem::transmute(a),
                mem::transmute(b),
                mem::transmute(c)
            ))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_mul_q_h() {
        #[rustfmt::skip]
        let a = i16x8::new(
            12500, -20, -300, 400,
            12500, 20, 300, 400
        );
        #[rustfmt::skip]
        let b = i16x8::new(
            1250, 10240, -7585, 8456,
            1250, 10240, -7585, 8456
        );
        #[rustfmt::skip]
        let r = i16x8::new(476, -7, 69, 103, 476, 6, -70, 103);

        assert_eq!(
            r,
            mem::transmute(__msa_mul_q_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_mul_q_w() {
        #[rustfmt::skip]
        let a = i32x4::new(
            i32::MAX, i32::MAX,
            i32::MIN, i32::MIN
        );
        #[rustfmt::skip]
        let b = i32x4::new(30, 60, 30, 60);
        #[rustfmt::skip]
        let r = i32x4::new(29, 59, -30, -60);

        assert_eq!(
            r,
            mem::transmute(__msa_mul_q_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_mulr_q_h() {
        #[rustfmt::skip]
        let a = i16x8::new(
            12500, -20, -300, 400,
            12500, 20, 300, 400
        );
        #[rustfmt::skip]
        let b = i16x8::new(
            1250, 10240, -7585, 8456,
            1250, 10240, -7585, 8456
        );
        #[rustfmt::skip]
        let r = i16x8::new(477, -6, 69, 103, 477, 6, -69, 103);

        assert_eq!(
            r,
            mem::transmute(__msa_mulr_q_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_mulr_q_w() {
        #[rustfmt::skip]
        let a = i32x4::new(
            i32::MAX, i32::MAX,
            i32::MIN, i32::MIN
        );
        #[rustfmt::skip]
        let b = i32x4::new(30, 60, 30, 60);
        #[rustfmt::skip]
        let r = i32x4::new(30, 60, -30, -60);

        assert_eq!(
            r,
            mem::transmute(__msa_mulr_q_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_mulv_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16
        );
        #[rustfmt::skip]
        let b = i8x16::new(
            16, 15, 14, 13,
            12, 11, 10, 9,
            8, 7, 6, 5,
            4, 3, 2, 1
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            16, 30, 42, 52,
            60, 66, 70, 72,
            72, 70, 66, 60,
            52, 42, 30, 16
        );

        assert_eq!(
            r,
            mem::transmute(__msa_mulv_b(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_mulv_h() {
        #[rustfmt::skip]
        let a = i16x8::new(
            1, 2, 3, 4,
            5, 6, 7, 8
        );
        #[rustfmt::skip]
        let b = i16x8::new(
            8, 7, 6, 5,
            4, 3, 2, 1
        );
        #[rustfmt::skip]
        let r = i16x8::new(8, 14, 18, 20, 20, 18, 14, 8);

        assert_eq!(
            r,
            mem::transmute(__msa_mulv_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_mulv_w() {
        #[rustfmt::skip]
        let a = i32x4::new(1, 2, 3, 4);
        #[rustfmt::skip]
        let b = i32x4::new(4, 3, 2, 1);
        #[rustfmt::skip]
        let r = i32x4::new(4, 6, 6, 4);

        assert_eq!(
            r,
            mem::transmute(__msa_mulv_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_mulv_d() {
        #[rustfmt::skip]
        let a = i64x2::new(1, 2);
        #[rustfmt::skip]
        let b = i64x2::new(2, 1);
        #[rustfmt::skip]
        let r = i64x2::new(2, 2);

        assert_eq!(
            r,
            mem::transmute(__msa_mulv_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_nloc_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            -128, -64, -32, -16,
            -8, -4, -2, -1,
            1, 2, 4, 8,
            16, 32, 64, 127
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            1, 2, 3, 4,
            5, 6, 7, 8,
            0, 0, 0, 0,
            0, 0, 0, 0
        );

        assert_eq!(r, mem::transmute(__msa_nloc_b(mem::transmute(a))));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_nloc_h() {
        #[rustfmt::skip]
        let a = i16x8::new(
            -32768, -16384, -8192, -4096,
            4096, 8192, 16384, 32767
        );
        #[rustfmt::skip]
        let r = i16x8::new(1, 2, 3, 4, 0, 0, 0, 0);

        assert_eq!(r, mem::transmute(__msa_nloc_h(mem::transmute(a))));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_nloc_w() {
        #[rustfmt::skip]
        let a = i32x4::new(
            i32::MIN, -1073741824,
            1073741824, i32::MAX
        );
        #[rustfmt::skip]
        let r = i32x4::new(1, 2, 0, 0);

        assert_eq!(r, mem::transmute(__msa_nloc_w(mem::transmute(a))));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_nloc_d() {
        #[rustfmt::skip]
        let a = i64x2::new(i64::MIN, i64::MAX);
        #[rustfmt::skip]
        let r = i64x2::new(1, 0);

        assert_eq!(r, mem::transmute(__msa_nloc_d(mem::transmute(a))));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_nlzc_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            7, 6, 6, 5,
            5, 5, 5, 4,
            4, 4, 4, 4,
            4, 4, 4, 3
        );

        assert_eq!(r, mem::transmute(__msa_nlzc_b(mem::transmute(a))));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_nlzc_h() {
        #[rustfmt::skip]
        let a = i16x8::new(
            1, 2, 3, 4,
            5, 6, 7, 8
        );
        #[rustfmt::skip]
        let r = i16x8::new(15, 14, 14, 13, 13, 13, 13, 12);

        assert_eq!(r, mem::transmute(__msa_nlzc_h(mem::transmute(a))));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_nlzc_w() {
        #[rustfmt::skip]
        let a = i32x4::new(1, 2, 3, 4);
        #[rustfmt::skip]
        let r = i32x4::new(31, 30, 30, 29);

        assert_eq!(r, mem::transmute(__msa_nlzc_w(mem::transmute(a))));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_nlzc_d() {
        #[rustfmt::skip]
        let a = i64x2::new(1, 2);
        #[rustfmt::skip]
        let r = i64x2::new(63, 62);

        assert_eq!(r, mem::transmute(__msa_nlzc_d(mem::transmute(a))));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_nor_v() {
        #[rustfmt::skip]
        let a = u8x16::new(
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16
        );
        #[rustfmt::skip]
        let b = u8x16::new(
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16
        );
        #[rustfmt::skip]
        let r = u8x16::new(
            254, 253, 252, 251,
            250, 249, 248, 247,
            246, 245, 244, 243,
            242, 241, 240, 239
        );

        assert_eq!(
            r,
            mem::transmute(__msa_nor_v(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_nori_b() {
        #[rustfmt::skip]
        let a = u8x16::new(
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16
        );
        #[rustfmt::skip]
        let r = u8x16::new(
            250, 249, 248, 251,
            250, 249, 248, 243,
            242, 241, 240, 243,
            242, 241, 240, 235
        );

        assert_eq!(r, mem::transmute(__msa_nori_b(mem::transmute(a), 4)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_or_v() {
        #[rustfmt::skip]
        let a = u8x16::new(
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16
        );
        #[rustfmt::skip]
        let b = u8x16::new(
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16
        );
        #[rustfmt::skip]
        let r = u8x16::new(
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16
        );

        assert_eq!(
            r,
            mem::transmute(__msa_or_v(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_ori_b() {
        #[rustfmt::skip]
        let a = u8x16::new(
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16
        );
        #[rustfmt::skip]
        let r = u8x16::new(
            5, 6, 7, 4,
            5, 6, 7, 12,
            13, 14, 15, 12,
            13, 14, 15, 20
        );

        assert_eq!(r, mem::transmute(__msa_ori_b(mem::transmute(a), 4)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_pckev_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            1, 2, 3, 4,
            1, 2, 3, 4,
            1, 2, 3, 4,
            1, 2, 3, 4
        );
        #[rustfmt::skip]
        let b = i8x16::new(
            4, 3, 2, 1,
            4, 3, 2, 1,
            4, 3, 2, 1,
            4, 3, 2, 1
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            4, 2, 4, 2,
            4, 2, 4, 2,
            1, 3, 1, 3,
            1, 3, 1, 3
        );

        assert_eq!(
            r,
            mem::transmute(__msa_pckev_b(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_pckev_h() {
        #[rustfmt::skip]
        let a = i16x8::new(1, 2, 3, 4, 1, 2, 3, 4);
        #[rustfmt::skip]
        let b = i16x8::new(4, 3, 2, 1, 4, 3, 2, 1);
        #[rustfmt::skip]
        let r = i16x8::new(4, 2, 4, 2, 1, 3, 1, 3);

        assert_eq!(
            r,
            mem::transmute(__msa_pckev_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_pckev_w() {
        #[rustfmt::skip]
        let a = i32x4::new(1, 2, 3, 4);
        #[rustfmt::skip]
        let b = i32x4::new(4, 3, 2, 1);
        #[rustfmt::skip]
        let r = i32x4::new(4, 2, 1, 3);

        assert_eq!(
            r,
            mem::transmute(__msa_pckev_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_pckev_d() {
        #[rustfmt::skip]
        let a = i64x2::new(1, 2);
        #[rustfmt::skip]
        let b = i64x2::new(4, 3);
        #[rustfmt::skip]
        let r = i64x2::new(4, 1);

        assert_eq!(
            r,
            mem::transmute(__msa_pckev_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_pckod_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            1, 2, 3, 4,
            1, 2, 3, 4,
            1, 2, 3, 4,
            1, 2, 3, 4
        );
        #[rustfmt::skip]
        let b = i8x16::new(
            4, 3, 2, 1,
            4, 3, 2, 1,
            4, 3, 2, 1,
            4, 3, 2, 1
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            3, 1, 3, 1,
            3, 1, 3, 1,
            2, 4, 2, 4,
            2, 4, 2, 4
        );

        assert_eq!(
            r,
            mem::transmute(__msa_pckod_b(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_pckod_h() {
        #[rustfmt::skip]
        let a = i16x8::new(1, 2, 3, 4, 1, 2, 3, 4);
        #[rustfmt::skip]
        let b = i16x8::new(4, 3, 2, 1, 4, 3, 2, 1);
        #[rustfmt::skip]
        let r = i16x8::new(3, 1, 3, 1, 2, 4, 2, 4);

        assert_eq!(
            r,
            mem::transmute(__msa_pckod_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_pckod_w() {
        #[rustfmt::skip]
        let a = i32x4::new(1, 2, 3, 4);
        #[rustfmt::skip]
        let b = i32x4::new(4, 3, 2, 1);
        #[rustfmt::skip]
        let r = i32x4::new(3, 1, 2, 4);

        assert_eq!(
            r,
            mem::transmute(__msa_pckod_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_pckod_d() {
        #[rustfmt::skip]
        let a = i64x2::new(1, 2);
        #[rustfmt::skip]
        let b = i64x2::new(4, 3);
        #[rustfmt::skip]
        let r = i64x2::new(3, 2);

        assert_eq!(
            r,
            mem::transmute(__msa_pckod_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_pcnt_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            -128, -64, -32, -16,
            -8, -4, -2, -1,
            1, 2, 4, 8,
            16, 32, 64, 127
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            1, 2, 3, 4,
            5, 6, 7, 8,
            1, 1, 1, 1,
            1, 1, 1, 7
        );

        assert_eq!(r, mem::transmute(__msa_pcnt_b(mem::transmute(a))));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_pcnt_h() {
        #[rustfmt::skip]
        let a = i16x8::new(
            -32768, -16384, -8192, -4096,
            4096, 8192, 16384, 32767
        );
        #[rustfmt::skip]
        let r = i16x8::new(1, 2, 3, 4, 1, 1, 1, 15);

        assert_eq!(r, mem::transmute(__msa_pcnt_h(mem::transmute(a))));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_pcnt_w() {
        #[rustfmt::skip]
        let a = i32x4::new(
            i32::MIN, -1073741824,
            1073741824, i32::MAX
        );
        #[rustfmt::skip]
        let r = i32x4::new(1, 2, 1, 31);

        assert_eq!(r, mem::transmute(__msa_pcnt_w(mem::transmute(a))));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_pcnt_d() {
        #[rustfmt::skip]
        let a = i64x2::new(-2147483648, 2147483647);
        #[rustfmt::skip]
        let r = i64x2::new(33, 31);

        assert_eq!(r, mem::transmute(__msa_pcnt_d(mem::transmute(a))));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_sat_s_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            i8::MAX, 105, 30, 1,
            i8::MAX, 105, 30, 1,
            i8::MAX, 105, 30, 1,
            i8::MAX, 105, 30, 1
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            3, 3, 3, 1,
            3, 3, 3, 1,
            3, 3, 3, 1,
            3, 3, 3, 1
        );

        assert_eq!(r, mem::transmute(__msa_sat_s_b(mem::transmute(a), 2)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_sat_s_h() {
        #[rustfmt::skip]
        let a = i16x8::new(
            i16::MAX, 1155, 155, 1,
            i16::MAX, 1155, 155, 1
        );
        #[rustfmt::skip]
        let r = i16x8::new(127, 127, 127, 1, 127, 127, 127, 1);

        assert_eq!(r, mem::transmute(__msa_sat_s_h(mem::transmute(a), 7)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_sat_s_w() {
        #[rustfmt::skip]
        let a = i32x4::new(i32::MAX, 111111155, i32::MAX, 1);
        #[rustfmt::skip]
        let r = i32x4::new(131071, 131071, 131071, 1);

        assert_eq!(r, mem::transmute(__msa_sat_s_w(mem::transmute(a), 17)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_sat_s_d() {
        #[rustfmt::skip]
        let a = i64x2::new(i64::MAX, 1);
        #[rustfmt::skip]
        let r = i64x2::new(137438953471, 1);

        assert_eq!(r, mem::transmute(__msa_sat_s_d(mem::transmute(a), 37)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_sat_u_b() {
        #[rustfmt::skip]
        let a = u8x16::new(
            u8::MAX, 105, 30, 1,
            u8::MAX, 105, 30, 1,
            u8::MAX, 105, 30, 1,
            u8::MAX, 105, 30, 1
        );
        #[rustfmt::skip]
        let r = u8x16::new(
            7, 7, 7, 1,
            7, 7, 7, 1,
            7, 7, 7, 1,
            7, 7, 7, 1
        );

        assert_eq!(r, mem::transmute(__msa_sat_u_b(mem::transmute(a), 2)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_sat_u_h() {
        #[rustfmt::skip]
        let a = u16x8::new(
            u16::MAX, 1155, 155, 1,
            u16::MAX, 1155, 155, 1
        );
        #[rustfmt::skip]
        let r = u16x8::new(255, 255, 155, 1, 255, 255, 155, 1);

        assert_eq!(r, mem::transmute(__msa_sat_u_h(mem::transmute(a), 7)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_sat_u_w() {
        #[rustfmt::skip]
        let a = u32x4::new(u32::MAX, 111111155, u32::MAX, 1);
        #[rustfmt::skip]
        let r = u32x4::new(262143, 262143, 262143, 1);

        assert_eq!(r, mem::transmute(__msa_sat_u_w(mem::transmute(a), 17)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_sat_u_d() {
        #[rustfmt::skip]
        let a = u64x2::new(u64::MAX, 1);
        #[rustfmt::skip]
        let r = u64x2::new(274877906943, 1);

        assert_eq!(r, mem::transmute(__msa_sat_u_d(mem::transmute(a), 37)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_shf_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            11, 12, 3, 4,
            11, 12, 3, 4,
            11, 12, 3, 4,
            11, 12, 3, 4
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            11, 3, 4, 12,
            11, 3, 4, 12,
            11, 3, 4, 12,
            11, 3, 4, 12
        );

        assert_eq!(r, mem::transmute(__msa_shf_b(mem::transmute(a), 120)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_shf_h() {
        #[rustfmt::skip]
        let a = i16x8::new(
            11, 12, 13, 14,
            11, 12, 13, 14
        );
        #[rustfmt::skip]
        let r = i16x8::new(11, 14, 12, 13, 11, 14, 12, 13);

        assert_eq!(r, mem::transmute(__msa_shf_h(mem::transmute(a), 156)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_shf_w() {
        #[rustfmt::skip]
        let a = i32x4::new(1, 2, 3, 4);
        #[rustfmt::skip]
        let r = i32x4::new(1, 3, 2, 4);

        assert_eq!(r, mem::transmute(__msa_shf_w(mem::transmute(a), 216)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_sld_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            0, 1, 2, 3,
            4, 5, 6, 7,
            8, 9, 10, 11,
            12, 13, 14, 15
        );
        #[rustfmt::skip]
        let b = i8x16::new(
            16, 17, 18, 19,
            20, 21, 22, 23,
            24, 25, 26, 27,
            28, 29, 30, 31
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            21, 22, 23, 24,
            25, 26, 27, 28,
            29, 30, 31, 0,
            1, 2, 3, 4
        );

        assert_eq!(
            r,
            mem::transmute(__msa_sld_b(mem::transmute(a), mem::transmute(b), 5))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_sld_h() {
        #[rustfmt::skip]
        let a = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        #[rustfmt::skip]
        let b = i16x8::new(8, 9, 10, 11, 12, 13, 14, 15);
        // let c = 5 as i32;
        let r = i16x8::new(9, 10, 11, 0, 13, 14, 15, 4);

        assert_eq!(
            r,
            mem::transmute(__msa_sld_h(mem::transmute(a), mem::transmute(b), 2))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_sld_w() {
        #[rustfmt::skip]
        let a = i32x4::new(0, 1, 2, 3);
        #[rustfmt::skip]
        let b = i32x4::new(4, 5, 6, 7);
        #[rustfmt::skip]
        let r = i32x4::new(4, 5, 6, 7);

        assert_eq!(
            r,
            mem::transmute(__msa_sld_w(mem::transmute(a), mem::transmute(b), 4))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_sld_d() {
        #[rustfmt::skip]
        let a = i64x2::new(0, 1);
        #[rustfmt::skip]
        let b = i64x2::new(2, 3);
        #[rustfmt::skip]
        let r = i64x2::new(2, 3);

        assert_eq!(
            r,
            mem::transmute(__msa_sld_d(mem::transmute(a), mem::transmute(b), 2))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_sldi_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            0, 1, 2, 3,
            4, 5, 6, 7,
            8, 9, 10, 11,
            12, 13, 14, 15
        );
        #[rustfmt::skip]
        let b = i8x16::new(
            16, 17, 18, 19,
            20, 21, 22, 23,
            24, 25, 26, 27,
            28, 29, 30, 31
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            21, 22, 23, 24,
            25, 26, 27, 28,
            29, 30, 31, 0,
            1, 2, 3, 4
        );

        assert_eq!(
            r,
            mem::transmute(__msa_sldi_b(mem::transmute(a), mem::transmute(b), 5))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_sldi_h() {
        #[rustfmt::skip]
        let a = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        #[rustfmt::skip]
        let b = i16x8::new(8, 9, 10, 11, 12, 13, 14, 15);
        // let c = 5 as i32;
        let r = i16x8::new(9, 10, 11, 0, 13, 14, 15, 4);

        assert_eq!(
            r,
            mem::transmute(__msa_sldi_h(mem::transmute(a), mem::transmute(b), 2))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_sldi_w() {
        #[rustfmt::skip]
        let a = i32x4::new(0, 1, 2, 3);
        #[rustfmt::skip]
        let b = i32x4::new(4, 5, 6, 7);
        #[rustfmt::skip]
        let r = i32x4::new(4, 5, 6, 7);

        assert_eq!(
            r,
            mem::transmute(__msa_sldi_w(mem::transmute(a), mem::transmute(b), 4))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_sldi_d() {
        #[rustfmt::skip]
        let a = i64x2::new(0, 1);
        #[rustfmt::skip]
        let b = i64x2::new(2, 3);
        #[rustfmt::skip]
        let r = i64x2::new(2, 3);

        assert_eq!(
            r,
            mem::transmute(__msa_sldi_d(mem::transmute(a), mem::transmute(b), 2))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_sll_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            1, 2, 3, 4,
            1, 2, 3, 4,
            1, 2, 3, 4,
            1, 2, 3, 4
        );
        #[rustfmt::skip]
        let b = i8x16::new(
            4, 3, 2, 1,
            4, 3, 2, 1,
            4, 3, 2, 1,
            4, 3, 2, 1
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            16, 16, 12, 8,
            16, 16, 12, 8,
            16, 16, 12, 8,
            16, 16, 12, 8
        );

        assert_eq!(
            r,
            mem::transmute(__msa_sll_b(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_sll_h() {
        #[rustfmt::skip]
        let a = i16x8::new(1, 2, 3, 4, 1, 2, 3, 4);
        #[rustfmt::skip]
        let b = i16x8::new(4, 3, 2, 1, 4, 3, 2, 1);
        #[rustfmt::skip]
        let r = i16x8::new(16, 16, 12, 8, 16, 16, 12, 8);

        assert_eq!(
            r,
            mem::transmute(__msa_sll_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_sll_w() {
        #[rustfmt::skip]
        let a = i32x4::new(1, 2, 3, 4);
        #[rustfmt::skip]
        let b = i32x4::new(4, 3, 2, 1);
        #[rustfmt::skip]
        let r = i32x4::new(16, 16, 12, 8);

        assert_eq!(
            r,
            mem::transmute(__msa_sll_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_sll_d() {
        #[rustfmt::skip]
        let a = i64x2::new(1, 2);
        #[rustfmt::skip]
        let b = i64x2::new(4, 3);
        #[rustfmt::skip]
        let r = i64x2::new(16, 16);

        assert_eq!(
            r,
            mem::transmute(__msa_sll_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_slli_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            1, 2, 3, 4,
            1, 2, 3, 4,
            1, 2, 3, 4,
            1, 2, 3, 4
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            4, 8, 12, 16,
            4, 8, 12, 16,
            4, 8, 12, 16,
            4, 8, 12, 16
        );

        assert_eq!(r, mem::transmute(__msa_slli_b(mem::transmute(a), 2)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_slli_h() {
        #[rustfmt::skip]
        let a = i16x8::new(
            1, 2, 3, 4,
            1, 2, 3, 4  
        );
        #[rustfmt::skip]
        let r = i16x8::new(4, 8, 12, 16, 4, 8, 12, 16);

        assert_eq!(r, mem::transmute(__msa_slli_h(mem::transmute(a), 2)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_slli_w() {
        #[rustfmt::skip]
        let a = i32x4::new(1, 2, 3, 4);
        #[rustfmt::skip]
        let r = i32x4::new(4, 8, 12, 16);

        assert_eq!(r, mem::transmute(__msa_slli_w(mem::transmute(a), 2)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_slli_d() {
        #[rustfmt::skip]
        let a = i64x2::new(1, 2);
        #[rustfmt::skip]
        let r = i64x2::new(2, 4);

        assert_eq!(r, mem::transmute(__msa_slli_d(mem::transmute(a), 1)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_splat_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            1, 2, 3, 4,
            1, 2, 3, 4,
            1, 2, 3, 4,
            1, 2, 3, 4
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            4, 4, 4, 4,
            4, 4, 4, 4,
            4, 4, 4, 4,
            4, 4, 4, 4
        );

        assert_eq!(r, mem::transmute(__msa_splat_b(mem::transmute(a), 3)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_splat_h() {
        #[rustfmt::skip]
        let a = i16x8::new(
            1, 2, 3, 4,
            1, 2, 3, 4,
        );
        #[rustfmt::skip]
        let r = i16x8::new(4, 4, 4, 4, 4, 4, 4, 4);

        assert_eq!(r, mem::transmute(__msa_splat_h(mem::transmute(a), 3)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_splat_w() {
        #[rustfmt::skip]
        let a = i32x4::new(1, 2, 3, 4);
        #[rustfmt::skip]
        let r = i32x4::new(4, 4, 4, 4);

        assert_eq!(r, mem::transmute(__msa_splat_w(mem::transmute(a), 3)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_splat_d() {
        #[rustfmt::skip]
        let a = i64x2::new(1, 2);
        #[rustfmt::skip]
        let r = i64x2::new(2, 2);

        assert_eq!(r, mem::transmute(__msa_splat_d(mem::transmute(a), 3)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_splati_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            1, 2, 3, 4,
            1, 2, 3, 4,
            1, 2, 3, 4,
            1, 2, 3, 4
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            3, 3, 3, 3,
            3, 3, 3, 3,
            3, 3, 3, 3,
            3, 3, 3, 3
        );

        assert_eq!(r, mem::transmute(__msa_splati_b(mem::transmute(a), 2)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_splati_h() {
        #[rustfmt::skip]
        let a = i16x8::new(
            1, 2, 3, 4,
            1, 2, 3, 4,
        );
        #[rustfmt::skip]
        let r = i16x8::new(3, 3, 3, 3, 3, 3, 3, 3);

        assert_eq!(r, mem::transmute(__msa_splati_h(mem::transmute(a), 2)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_splati_w() {
        #[rustfmt::skip]
        let a = i32x4::new(1, 2, 3, 4);
        #[rustfmt::skip]
        let r = i32x4::new(3, 3, 3, 3);

        assert_eq!(r, mem::transmute(__msa_splati_w(mem::transmute(a), 2)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_splati_d() {
        #[rustfmt::skip]
        let a = i64x2::new(1, 2);
        #[rustfmt::skip]
        let r = i64x2::new(2, 2);

        assert_eq!(r, mem::transmute(__msa_splati_d(mem::transmute(a), 1)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_sra_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            -128, -64, -32, -16,
            -8, -4, -2, -1,
            1, 2, 4, 8,
            16, 32, 64, 127
        );
        #[rustfmt::skip]
        let b = i8x16::new(
            8, 7, 6, 5,
            4, 3, 2, 1,
            8, 7, 6, 5,
            4, 3, 2, 1
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            -128, -1, -1, -1,
            -1, -1, -1, -1,
            1, 0, 0, 0,
            1, 4, 16, 63
        );

        assert_eq!(
            r,
            mem::transmute(__msa_sra_b(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_sra_h() {
        #[rustfmt::skip]
        let a = i16x8::new(
            -32768, -16384, -8192, -4096,
            1, 2, 3, 4
        );
        #[rustfmt::skip]
        let b = i16x8::new(
            15, 14, 13, 12,
            12, 13, 14, 15
        );
        #[rustfmt::skip]
        let r = i16x8::new(
            -1, -1, -1, -1,
            0, 0, 0, 0
        );

        assert_eq!(
            r,
            mem::transmute(__msa_sra_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_sra_w() {
        #[rustfmt::skip]
        let a = i32x4::new(i32::MIN, -1073741824, 1, 2);
        #[rustfmt::skip]
        let b = i32x4::new(16, 15, 16, 15);
        #[rustfmt::skip]
        let r = i32x4::new(-32768, -32768, 0, 0);

        assert_eq!(
            r,
            mem::transmute(__msa_sra_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_sra_d() {
        #[rustfmt::skip]
        let a = i64x2::new(i64::MIN, i64::MAX);
        #[rustfmt::skip]
        let b = i64x2::new(32, 31);
        #[rustfmt::skip]
        let r = i64x2::new(-2147483648, 4294967295);

        assert_eq!(
            r,
            mem::transmute(__msa_sra_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_srai_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            i8::MAX, 125, 55, 1,
            i8::MAX, 125, 55, 1,
            i8::MAX, 125, 55, 1,
            i8::MAX, 125, 55, 1
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            31, 31, 13, 0,
            31, 31, 13, 0,
            31, 31, 13, 0,
            31, 31, 13, 0
        );

        assert_eq!(r, mem::transmute(__msa_srai_b(mem::transmute(a), 2)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_srai_h() {
        #[rustfmt::skip]
        let a = i16x8::new(
            i16::MAX, 125, 55, 1, 
            i16::MAX, 125, 55, 1
        );
        #[rustfmt::skip]
        let r = i16x8::new(8191, 31, 13, 0, 8191, 31, 13, 0);

        assert_eq!(r, mem::transmute(__msa_srai_h(mem::transmute(a), 2)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_srai_w() {
        #[rustfmt::skip]
        let a = i32x4::new(i32::MAX, 125, 55, 1);
        let r = i32x4::new(536870911, 31, 13, 0);

        assert_eq!(r, mem::transmute(__msa_srai_w(mem::transmute(a), 2)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_srai_d() {
        #[rustfmt::skip]
        let a = i64x2::new(i64::MAX, 55);
        #[rustfmt::skip]
        let r = i64x2::new(2305843009213693951, 13);

        assert_eq!(r, mem::transmute(__msa_srai_d(mem::transmute(a), 2)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_srar_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            -128, -64, -32, -16,
            -8, -4, -2, -1,
            1, 2, 4, 8,
            16, 32, 64, 127
        );
        #[rustfmt::skip]
        let b = i8x16::new(
            4, 3, 2, 1,
            4, 3, 2, 1,
            8, 7, 6, 5,
            4, 3, 2, 1
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            -8, -8, -8, -8,
            0, 0, 0, 0,
            1, 0, 0, 0,
            1, 4, 16, 64
        );

        assert_eq!(
            r,
            mem::transmute(__msa_srar_b(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_srar_h() {
        #[rustfmt::skip]
        let a = i16x8::new(
            i16::MIN, -16384, -8192, -4096,
            150, 50, 25, 15
        );
        #[rustfmt::skip]
        let b = i16x8::new(
            4, 3, 2, 1,
            1, 2, 3, 4
        );
        #[rustfmt::skip]
        let r = i16x8::new(
            -2048, -2048, -2048, -2048, 
            75, 13, 3, 1
        );

        assert_eq!(
            r,
            mem::transmute(__msa_srar_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_srar_w() {
        #[rustfmt::skip]
        let a = i32x4::new(i32::MIN, -1073741824, 100, 50);
        #[rustfmt::skip]
        let b = i32x4::new(16, 15, 1, 2);
        #[rustfmt::skip]
        let r = i32x4::new(-32768, -32768, 50, 13);

        assert_eq!(
            r,
            mem::transmute(__msa_srar_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_srar_d() {
        #[rustfmt::skip]
        let a = i64x2::new(i64::MIN, i64::MAX);
        #[rustfmt::skip]
        let b = i64x2::new(32, 31);
        #[rustfmt::skip]
        let r = i64x2::new(-2147483648, 4294967296);

        assert_eq!(
            r,
            mem::transmute(__msa_srar_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_srari_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            125, i8::MAX, 55, 1,
            125, i8::MAX, 55, 1,
            125, i8::MAX, 55, 1,
            125, i8::MAX, 55, 1
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            31, 32, 14, 0,
            31, 32, 14, 0,
            31, 32, 14, 0,
            31, 32, 14, 0
        );

        assert_eq!(r, mem::transmute(__msa_srari_b(mem::transmute(a), 2)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_srari_h() {
        #[rustfmt::skip]
        let a = i16x8::new(2155, 1155, 155, 1, 2155, 1155, 155, 1);
        #[rustfmt::skip]
        let r = i16x8::new(539, 289, 39, 0, 539, 289, 39, 0);

        assert_eq!(r, mem::transmute(__msa_srari_h(mem::transmute(a), 2)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_srari_w() {
        #[rustfmt::skip]
        let a = i32x4::new(211111155, 111111155, 11111155, 1);
        #[rustfmt::skip]
        let r = i32x4::new(52777789, 27777789, 2777789, 0);

        assert_eq!(r, mem::transmute(__msa_srari_w(mem::transmute(a), 2)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_srari_d() {
        #[rustfmt::skip]
        let a = i64x2::new(211111111155, 111111111155);
        #[rustfmt::skip]
        let r = i64x2::new(52777777789, 27777777789);

        assert_eq!(r, mem::transmute(__msa_srari_d(mem::transmute(a), 2)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_srl_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            -128, -64, -32, -16,
            -8, -4, -2, -1,
            1, 2, 4, 8,
            16, 32, 64, 127
        );
        #[rustfmt::skip]
        let b = i8x16::new(
            8, 7, 6, 5,
            4, 3, 2, 1,
            8, 7, 6, 5,
            4, 3, 2, 1
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            -128, 1, 3, 7,
            15, 31, 63, 127,
            1, 0, 0, 0,
            1, 4, 16, 63
        );

        assert_eq!(
            r,
            mem::transmute(__msa_srl_b(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_srl_h() {
        #[rustfmt::skip]
        let a = i16x8::new(
            -32768, -16384, -8192, -4096,
            1, 2, 3, 4
        );
        #[rustfmt::skip]
        let b = i16x8::new(
            15, 14, 13, 12,
            4, 3, 2, 1
        );
        #[rustfmt::skip]
        let r = i16x8::new(1, 3, 7, 15, 0, 0, 0, 2);

        assert_eq!(
            r,
            mem::transmute(__msa_srl_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_srl_w() {
        #[rustfmt::skip]
        let a = i32x4::new(i32::MIN, -1073741824, 1, 2);
        #[rustfmt::skip]
        let b = i32x4::new(16, 15, 16, 15);
        #[rustfmt::skip]
        let r = i32x4::new(32768, 98304, 0, 0);

        assert_eq!(
            r,
            mem::transmute(__msa_srl_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_srl_d() {
        #[rustfmt::skip]
        let a = i64x2::new(i64::MIN, i64::MAX);
        #[rustfmt::skip]
        let b = i64x2::new(32, 31);
        #[rustfmt::skip]
        let r = i64x2::new(2147483648, 4294967295);

        assert_eq!(
            r,
            mem::transmute(__msa_srl_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_srli_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            25, 50, 100, 127,
            25, 50, 100, 127,
            25, 50, 100, 127,
            25, 50, 100, 127
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            6, 12, 25, 31,
            6, 12, 25, 31, 
            6, 12, 25, 31, 
            6, 12, 25, 31
        );

        assert_eq!(r, mem::transmute(__msa_srli_b(mem::transmute(a), 2)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_srli_h() {
        #[rustfmt::skip]
        let a = i16x8::new(
            i16::MAX, 3276, 100, 127,
            i16::MAX, 3276, 100, 127
        );
        #[rustfmt::skip]
        let r = i16x8::new(
            8191, 819, 25, 31,
            8191, 819, 25, 31
        );

        assert_eq!(r, mem::transmute(__msa_srli_h(mem::transmute(a), 2)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_srli_w() {
        #[rustfmt::skip]
        let a = i32x4::new(100, i32::MAX, 100, i32::MAX);
        #[rustfmt::skip]
        let r = i32x4::new(25, 536870911, 25, 536870911);

        assert_eq!(r, mem::transmute(__msa_srli_w(mem::transmute(a), 2)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_srli_d() {
        #[rustfmt::skip]
        let a = i64x2::new(100, i64::MAX);
        #[rustfmt::skip]
        let r = i64x2::new(50, 4611686018427387903);

        assert_eq!(r, mem::transmute(__msa_srli_d(mem::transmute(a), 1)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_srlr_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            -128, -64, -32, -16,
            -8, -4, -2, -1,
            1, 2, 4, 8,
            16, 32, 64, 127
        );
        #[rustfmt::skip]
        let b = i8x16::new(
            8, 7, 6, 5,
            4, 3, 2, 1,
            8, 7, 6, 5,
            4, 3, 2, 1
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            -128, 2, 4, 8,
            16, 32, 64, -128,
            1, 0, 0, 0,
            1, 4, 16, 64
        );

        assert_eq!(
            r,
            mem::transmute(__msa_srlr_b(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_srlr_h() {
        #[rustfmt::skip]
        let a = i16x8::new(
            -32768, -16384, -8192, -4096,
            1, 2, 3, 4
        );
        #[rustfmt::skip]
        let b = i16x8::new(
            15, 14, 13, 12,
            4, 3, 2, 1
        );
        #[rustfmt::skip]
        let r = i16x8::new(1, 3, 7, 15, 0, 0, 1, 2);

        assert_eq!(
            r,
            mem::transmute(__msa_srlr_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_srlr_w() {
        #[rustfmt::skip]
        let a = i32x4::new(i32::MIN, -1073741824, 1, 2);
        #[rustfmt::skip]
        let b = i32x4::new(16, 15, 16, 15);
        let r = i32x4::new(32768, 98304, 0, 0);

        assert_eq!(
            r,
            mem::transmute(__msa_srlr_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_srlr_d() {
        #[rustfmt::skip]
        let a = i64x2::new(i64::MIN, i64::MAX);
        #[rustfmt::skip]
        let b = i64x2::new(32, 31);
        #[rustfmt::skip]
        let r = i64x2::new(2147483648, 4294967296);

        assert_eq!(
            r,
            mem::transmute(__msa_srlr_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_srlri_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            25, 50, 100, i8::MAX,
            25, 50, 100, i8::MAX,
            25, 50, 100, i8::MAX,
            25, 50, 100, i8::MAX
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            6, 13, 25, 32, 
            6, 13, 25, 32, 
            6, 13, 25, 32, 
            6, 13, 25, 32
        );

        assert_eq!(r, mem::transmute(__msa_srlri_b(mem::transmute(a), 2)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_srlri_h() {
        #[rustfmt::skip]
        let a = i16x8::new(
            i16::MAX, 3276, 100, 127,
            i16::MAX, 3276, 100, 127
        );
        let r = i16x8::new(8192, 819, 25, 32, 8192, 819, 25, 32);

        assert_eq!(r, mem::transmute(__msa_srlri_h(mem::transmute(a), 2)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_srlri_w() {
        #[rustfmt::skip]
        let a = i32x4::new(100, 150, 200, i32::MAX);
        #[rustfmt::skip]
        let r = i32x4::new(25, 38, 50, 536870912);

        assert_eq!(r, mem::transmute(__msa_srlri_w(mem::transmute(a), 2)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_srlri_d() {
        #[rustfmt::skip]
        let a = i64x2::new(100, i64::MAX);
        #[rustfmt::skip]
        let r = i64x2::new(50, 4611686018427387904);

        assert_eq!(r, mem::transmute(__msa_srlri_d(mem::transmute(a), 1)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_st_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            13, 14, 15, 16, 
            17, 18, 19, 20, 
            21, 22, 23, 24, 
            25, 26, 27, 28
        );
        #[rustfmt::skip]
        let mut arr : [i8; 16] = [
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0
        ];
        #[rustfmt::skip]
        let r : [i8; 16] = [
            13, 14, 15, 16, 
            17, 18, 19, 20, 
            21, 22, 23, 24, 
            25, 26, 27, 28
        ];
        __msa_st_b(mem::transmute(a), arr.as_mut_ptr() as *mut u8, 0);
        assert_eq!(arr, r);
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_st_h() {
        #[rustfmt::skip]
        let a = i16x8::new(13, 14, 15, 16, 17, 18, 19, 20);
        let mut arr: [i16; 8] = [0, 0, 0, 0, 0, 0, 0, 0];
        #[rustfmt::skip]
        let r  : [i16; 8] = [13, 14, 15, 16, 17, 18, 19, 20];
        __msa_st_h(mem::transmute(a), arr.as_mut_ptr() as *mut u8, 0);
        assert_eq!(arr, r);
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_st_w() {
        #[rustfmt::skip]
        let a = i32x4::new(13, 14, 15, 16);
        let mut arr: [i32; 4] = [0, 0, 0, 0];
        #[rustfmt::skip]
        let r  : [i32; 4] = [13, 14, 15, 16];
        __msa_st_w(mem::transmute(a), arr.as_mut_ptr() as *mut u8, 0);
        assert_eq!(arr, r);
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_st_d() {
        #[rustfmt::skip]
        let a = i64x2::new(13, 14);
        let mut arr: [i64; 2] = [0, 0];
        #[rustfmt::skip]
        let r : [i64; 2] = [13, 14];
        __msa_st_d(mem::transmute(a), arr.as_mut_ptr() as *mut u8, 0);
        assert_eq!(arr, r);
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_subs_s_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            i8::MIN, -2, -3, -4,
            i8::MIN, -2, -3, -4,
            i8::MIN, -2, -3, -4,
            i8::MIN, -2, -3, -4
        );
        #[rustfmt::skip]
        let b = i8x16::new(
            6, -7, 8, -9,
            6, -7, 8, -9,
            6, -7, 8, -9,
            6, -7, 8, -9
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            i8::MIN, 5, -11, 5,
            i8::MIN, 5, -11, 5,
            i8::MIN, 5, -11, 5,
            i8::MIN, 5, -11, 5
        );

        assert_eq!(
            r,
            mem::transmute(__msa_subs_s_b(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_subs_s_h() {
        #[rustfmt::skip]
        let a = i16x8::new(
            i16::MIN, -2, -3, -4,
            i16::MIN, -2, -3, -4
        );
        #[rustfmt::skip]
        let b = i16x8::new(6, -7, 8, -9, 6, -7, 8, -9);
        #[rustfmt::skip]
        let r = i16x8::new(
            i16::MIN, 5, -11, 5,
            i16::MIN, 5, -11, 5
        );

        assert_eq!(
            r,
            mem::transmute(__msa_subs_s_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_subs_s_w() {
        #[rustfmt::skip]
        let a = i32x4::new(i32::MIN, -2, -3, -4);
        #[rustfmt::skip]
        let b = i32x4::new(6, -7, 8, -9);
        #[rustfmt::skip]
        let r = i32x4::new(i32::MIN, 5, -11, 5);

        assert_eq!(
            r,
            mem::transmute(__msa_subs_s_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_subs_s_d() {
        #[rustfmt::skip]
        let a = i64x2::new(i64::MIN, -2);
        #[rustfmt::skip]
        let b = i64x2::new(6, -7);
        #[rustfmt::skip]
        let r = i64x2::new(i64::MIN, 5);

        assert_eq!(
            r,
            mem::transmute(__msa_subs_s_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_subs_u_b() {
        #[rustfmt::skip]
        let a = u8x16::new(
            u8::MAX, 2, 3, 4,
            u8::MAX, 2, 3, 4,
            u8::MAX, 2, 3, 4,
            u8::MAX, 2, 3, 4
        );
        #[rustfmt::skip]
        let b = u8x16::new(
            6, 7, 8, 9,
            6, 7, 8, 9,
            6, 7, 8, 9,
            6, 7, 8, 9
        );
        #[rustfmt::skip]
        let r = u8x16::new(
            249, 0, 0, 0,
            249, 0, 0, 0,
            249, 0, 0, 0,
            249, 0, 0, 0
        );

        assert_eq!(
            r,
            mem::transmute(__msa_subs_u_b(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_subs_u_h() {
        #[rustfmt::skip]
        let a = u16x8::new(
            u16::MAX, 2, 3, 4, 
            u16::MAX, 2, 3, 4
        );
        #[rustfmt::skip]
        let b = u16x8::new(6, 7, 8, 9, 6, 7, 8, 9);
        #[rustfmt::skip]
        let r = u16x8::new(65529, 0, 0, 0, 65529, 0, 0, 0);

        assert_eq!(
            r,
            mem::transmute(__msa_subs_u_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_subs_u_w() {
        #[rustfmt::skip]
        let a = u32x4::new(u32::MAX, 2, 3, 4);
        #[rustfmt::skip]
        let b = u32x4::new(6, 7, 8, 9);
        #[rustfmt::skip]
        let r = u32x4::new(4294967289, 0, 0, 0);

        assert_eq!(
            r,
            mem::transmute(__msa_subs_u_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_subs_u_d() {
        #[rustfmt::skip]
        let a = u64x2::new(u64::MAX, 2);
        #[rustfmt::skip]
        let b = u64x2::new(6, 7);
        #[rustfmt::skip]
        let r = u64x2::new(18446744073709551609, 0);

        assert_eq!(
            r,
            mem::transmute(__msa_subs_u_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_subsus_u_b() {
        #[rustfmt::skip]
        let a = u8x16::new(
            u8::MAX, 2, 3, 4,
            u8::MAX, 2, 3, 4,
            u8::MAX, 2, 3, 4,
            u8::MAX, 2, 3, 4
        );
        #[rustfmt::skip]
        let b = i8x16::new(
            -6, -7, -8, -9,
            -6, -7, -8, -9,
            -6, -7, -8, -9,
            -6, -7, -8, -9
        );
        #[rustfmt::skip]
        let r = u8x16::new(
            255, 9, 11, 13,
            255, 9, 11, 13,
            255, 9, 11, 13,
            255, 9, 11, 13
        );

        assert_eq!(
            r,
            mem::transmute(__msa_subsus_u_b(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_subsus_u_h() {
        #[rustfmt::skip]
        let a = u16x8::new(
            u16::MAX, 2, 3, 4,
            u16::MAX, 2, 3, 4
        );
        #[rustfmt::skip]
        let b = i16x8::new(-6, -7, -8, -9, -6, -7, -8, -9);
        #[rustfmt::skip]
        let r = u16x8::new(65535, 9, 11, 13, 65535, 9, 11, 13);

        assert_eq!(
            r,
            mem::transmute(__msa_subsus_u_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_subsus_u_w() {
        #[rustfmt::skip]
        let a = u32x4::new(u32::MAX, 2, 3, 4);
        #[rustfmt::skip]
        let b = i32x4::new(-6, -7, -8, -9);
        #[rustfmt::skip]
        let r = u32x4::new(4294967295, 9, 11, 13);

        assert_eq!(
            r,
            mem::transmute(__msa_subsus_u_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_subsus_u_d() {
        #[rustfmt::skip]
        let a = u64x2::new(u64::MAX, 2);
        #[rustfmt::skip]
        let b = i64x2::new(-6, -7);
        #[rustfmt::skip]
        let r = u64x2::new(18446744073709551615, 9);

        assert_eq!(
            r,
            mem::transmute(__msa_subsus_u_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_subsuu_s_b() {
        #[rustfmt::skip]
        let a = u8x16::new(
            u8::MAX, 2, 3, 4,
            u8::MAX, 2, 3, 4,
            u8::MAX, 2, 3, 4,
            u8::MAX, 2, 3, 4
        );
        #[rustfmt::skip]
        let b = u8x16::new(
            6, 7, 8, u8::MAX,
            6, 7, 8, u8::MAX,
            6, 7, 8, u8::MAX,
            6, 7, 8, u8::MAX
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            127, -5, -5, -128,
            127, -5, -5, -128,
            127, -5, -5, -128,
            127, -5, -5, -128
        );

        assert_eq!(
            r,
            mem::transmute(__msa_subsuu_s_b(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_subsuu_s_h() {
        #[rustfmt::skip]
        let a = u16x8::new(
            u16::MAX, 2, 3, 
            4, u16::MAX, 2, 3, 4
        );
        #[rustfmt::skip]
        let b = u16x8::new(6, 7, 8, 65535, 6, 7, 8, 65535);
        #[rustfmt::skip]
        let r = i16x8::new(32767, -5, -5, -32768, 32767, -5, -5, -32768);

        assert_eq!(
            r,
            mem::transmute(__msa_subsuu_s_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_subsuu_s_w() {
        #[rustfmt::skip]
        let a = u32x4::new(u32::MAX, 2, 3, 4);
        #[rustfmt::skip]
        let b = u32x4::new(6, 7, 8, 4294967295);
        #[rustfmt::skip]
        let r = i32x4::new(2147483647, -5, -5, -2147483648);

        assert_eq!(
            r,
            mem::transmute(__msa_subsuu_s_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_subsuu_s_d() {
        #[rustfmt::skip]
        let a = u64x2::new(u64::MAX, 2);
        #[rustfmt::skip]
        let b = u64x2::new(6, 7);
        #[rustfmt::skip]
        let r = i64x2::new(i64::MAX, -5);

        assert_eq!(
            r,
            mem::transmute(__msa_subsuu_s_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_subv_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            i8::MIN, -2, -3, -4,
            i8::MIN, -2, -3, -4,
            i8::MIN, -2, -3, -4,
            i8::MIN, -2, -3, -4
        );
        #[rustfmt::skip]
        let b = i8x16::new(
            6, -7, 8, -9,
            6, -7, 8, -9,
            6, -7, 8, -9,
            6, -7, 8, -9
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            122, 5, -11, 5,
            122, 5, -11, 5,
            122, 5, -11, 5,
            122, 5, -11, 5
        );

        assert_eq!(
            r,
            mem::transmute(__msa_subv_b(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_subv_h() {
        #[rustfmt::skip]
        let a = i16x8::new(
            i16::MIN, -2, -3, -4,
            i16::MIN, -2, -3, -4
        );
        #[rustfmt::skip]
        let b = i16x8::new(6, -7, 8, -9, 6, -7, 8, -9);
        #[rustfmt::skip]
        let r = i16x8::new(32762, 5, -11, 5, 32762, 5, -11, 5);

        assert_eq!(
            r,
            mem::transmute(__msa_subv_h(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_subv_w() {
        #[rustfmt::skip]
        let a = i32x4::new(i32::MIN, -2, -3, -4);
        #[rustfmt::skip]
        let b = i32x4::new(6, -7, 8, -9);
        #[rustfmt::skip]
        let r = i32x4::new(2147483642, 5, -11, 5);

        assert_eq!(
            r,
            mem::transmute(__msa_subv_w(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_subv_d() {
        #[rustfmt::skip]
        let a = i64x2::new(i64::MAX, -2);
        #[rustfmt::skip]
        let b = i64x2::new(6, -7);
        #[rustfmt::skip]
        let r = i64x2::new(9223372036854775801, 5);

        assert_eq!(
            r,
            mem::transmute(__msa_subv_d(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_subvi_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            100, i8::MAX, 50, i8::MIN,
            100, i8::MAX, 50, i8::MIN,
            100, i8::MAX, 50, i8::MIN,
            100, i8::MAX, 50, i8::MIN
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            95, 122, 45, 123,
            95, 122, 45, 123,
            95, 122, 45, 123,
            95, 122, 45, 123
        );

        assert_eq!(r, mem::transmute(__msa_subvi_b(mem::transmute(a), 5)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_subvi_h() {
        #[rustfmt::skip]
        let a = i16x8::new(
            i16::MAX, 3276, -100, i16::MIN,
            i16::MAX, 3276, -100, i16::MIN
        );
        #[rustfmt::skip]
        let r = i16x8::new(
            32762, 3271, -105, 32763,
            32762, 3271, -105, 32763
        );

        assert_eq!(r, mem::transmute(__msa_subvi_h(mem::transmute(a), 5)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_subvi_w() {
        #[rustfmt::skip]
        let a = i32x4::new(100, 150, 200, i32::MAX);
        #[rustfmt::skip]
        let r = i32x4::new(95, 145, 195, 2147483642);

        assert_eq!(r, mem::transmute(__msa_subvi_w(mem::transmute(a), 5)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_subvi_d() {
        #[rustfmt::skip]
        let a = i64x2::new(100, i64::MAX);
        #[rustfmt::skip]
        let r = i64x2::new(95, 9223372036854775802);

        assert_eq!(r, mem::transmute(__msa_subvi_d(mem::transmute(a), 5)));
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_vshf_b() {
        #[rustfmt::skip]
        let a = i8x16::new(
            1, 2, 3, 4,
            1, 2, 3, 4,
            1, 2, 3, 4,
            1, 2, 3, 4
        );
        #[rustfmt::skip]
        let b = i8x16::new(
            4, 3, 2, 1,
            4, 3, 2, 1,
            4, 3, 2, 1,
            4, 3, 2, 1
        );
        #[rustfmt::skip]
        let c = i8x16::new(
            4, 3, 2, 1,
            4, 3, 2, 1,
            4, 3, 2, 1,
            4, 3, 2, 1
        );
        #[rustfmt::skip]
        let r = i8x16::new(
            3, 2, 1, 4, 
            3, 2, 1, 4, 
            3, 2, 1, 4, 
            3, 2, 1, 4
        );

        assert_eq!(
            r,
            mem::transmute(__msa_vshf_b(
                mem::transmute(a),
                mem::transmute(b),
                mem::transmute(c)
            ))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_vshf_h() {
        #[rustfmt::skip]
        let a = i16x8::new(
            1, 2, 3, 4,
            1, 2, 3, 4
        );
        #[rustfmt::skip]
        let b = i16x8::new(
            4, 3, 2, 1,
            4, 3, 2, 1
        );
        #[rustfmt::skip]
        let c = i16x8::new(
            4, 3, 2, 1,
            4, 3, 2, 1
        );
        let r = i16x8::new(3, 2, 1, 4, 3, 2, 1, 4);

        assert_eq!(
            r,
            mem::transmute(__msa_vshf_h(
                mem::transmute(a),
                mem::transmute(b),
                mem::transmute(c)
            ))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_vshf_w() {
        #[rustfmt::skip]
        let a = i32x4::new(1, 2, 3, 4);
        #[rustfmt::skip]
        let b = i32x4::new(4, 3, 2, 1);
        #[rustfmt::skip]
        let c = i32x4::new(4, 3, 2, 1);
        #[rustfmt::skip]
        let r = i32x4::new(3, 2, 1, 4);

        assert_eq!(
            r,
            mem::transmute(__msa_vshf_w(
                mem::transmute(a),
                mem::transmute(b),
                mem::transmute(c)
            ))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_vshf_d() {
        #[rustfmt::skip]
        let a = i64x2::new(1, 2);
        #[rustfmt::skip]
        let b = i64x2::new(4, 3);
        #[rustfmt::skip]
        let c = i64x2::new(4, 3);
        #[rustfmt::skip]
        let r = i64x2::new(3, 4);

        assert_eq!(
            r,
            mem::transmute(__msa_vshf_d(
                mem::transmute(a),
                mem::transmute(b),
                mem::transmute(c)
            ))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_xor_v() {
        #[rustfmt::skip]
        let a = u8x16::new(
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16
        );
        #[rustfmt::skip]
        let b = u8x16::new(
            16, 15, 14, 13,
            12, 11, 10, 9,
            8, 7, 6, 5,
            4, 3, 2, 1
        );
        #[rustfmt::skip]
        let r = u8x16::new(
            17, 13, 13, 9, 
            9, 13, 13, 1, 
            1, 13, 13, 9, 
            9, 13, 13, 17
        );

        assert_eq!(
            r,
            mem::transmute(__msa_xor_v(mem::transmute(a), mem::transmute(b)))
        );
    }

    #[simd_test(enable = "msa")]
    unsafe fn test_msa_xori_b() {
        #[rustfmt::skip]
        let a = u8x16::new(
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16
        );
        #[rustfmt::skip]
        let r = u8x16::new(
            5, 6, 7, 0, 
            1, 2, 3, 12, 
            13, 14, 15, 8, 
            9, 10, 11, 20
        );

        assert_eq!(r, mem::transmute(__msa_xori_b(mem::transmute(a), 4)));
    }
}
