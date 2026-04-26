//! Hexagon HVX 64-byte vector mode intrinsics
//!
//! This module provides intrinsics for the Hexagon Vector Extensions (HVX)
//! in 64-byte vector mode (512-bit vectors).
//!
//! HVX is a wide vector extension designed for high-performance signal processing.
//! [Hexagon HVX Programmer's Reference Manual](https://docs.qualcomm.com/doc/80-N2040-61)
//!
//! ## Vector Types
//!
//! In 64-byte mode:
//! - `HvxVector` is 512 bits (64 bytes) containing 16 x 32-bit values
//! - `HvxVectorPair` is 1024 bits (128 bytes)
//! - `HvxVectorPred` is 512 bits (64 bytes) for predicate operations
//!
//! To use this module, compile with `-C target-feature=+hvx-length64b`.
//!
//! ## Naming Convention
//!
//! Function names preserve the original Q6 naming case because the convention
//! uses case to distinguish register types:
//! - `W` (uppercase) = vector pair (`HvxVectorPair`)
//! - `V` (uppercase) = vector (`HvxVector`)
//! - `Q` (uppercase) = predicate (`HvxVectorPred`)
//! - `R` = scalar register (`i32`)
//!
//! For example, `Q6_W_vcombine_VV` operates on a vector pair while
//! `Q6_V_hi_W` extracts a vector from a pair.
//!
//! ## Architecture Versions
//!
//! Different intrinsics require different HVX architecture versions. Use the
//! appropriate target feature to enable the required version:
//! - HVX v60: `-C target-feature=+hvxv60` (basic HVX operations)
//! - HVX v62: `-C target-feature=+hvxv62`
//! - HVX v65: `-C target-feature=+hvxv65` (includes floating-point support)
//! - HVX v66: `-C target-feature=+hvxv66`
//! - HVX v68: `-C target-feature=+hvxv68`
//! - HVX v69: `-C target-feature=+hvxv69`
//! - HVX v73: `-C target-feature=+hvxv73`
//! - HVX v79: `-C target-feature=+hvxv79`
//!
//! Each version includes all features from previous versions.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

#[cfg(test)]
use stdarch_test::assert_instr;

use crate::intrinsics::simd::{simd_add, simd_and, simd_or, simd_sub, simd_xor};

// HVX type definitions for 64-byte vector mode
types! {
    #![unstable(feature = "stdarch_hexagon", issue = "151523")]

    /// HVX vector type (512 bits / 64 bytes)
    ///
    /// This type represents a single HVX vector register containing 16 x 32-bit values.
    pub struct HvxVector(16 x i32);

    /// HVX vector pair type (1024 bits / 128 bytes)
    ///
    /// This type represents a pair of HVX vector registers, often used for
    /// operations that produce double-width results.
    pub struct HvxVectorPair(32 x i32);

    /// HVX vector predicate type (512 bits / 64 bytes)
    ///
    /// This type represents a predicate vector used for conditional operations.
    /// Each bit corresponds to a lane in the vector.
    pub struct HvxVectorPred(16 x i32);
}

// LLVM intrinsic declarations for 64-byte vector mode
#[allow(improper_ctypes)]
unsafe extern "unadjusted" {
    #[link_name = "llvm.hexagon.V6.extractw"]
    fn extractw(_: HvxVector, _: i32) -> i32;
    #[link_name = "llvm.hexagon.V6.get.qfext"]
    fn get_qfext(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.hi"]
    fn hi(_: HvxVectorPair) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.lo"]
    fn lo(_: HvxVectorPair) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.lvsplatb"]
    fn lvsplatb(_: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.lvsplath"]
    fn lvsplath(_: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.lvsplatw"]
    fn lvsplatw(_: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.pred.and"]
    fn pred_and(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.pred.and.n"]
    fn pred_and_n(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.pred.not"]
    fn pred_not(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.pred.or"]
    fn pred_or(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.pred.or.n"]
    fn pred_or_n(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.pred.scalar2"]
    fn pred_scalar2(_: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.pred.scalar2v2"]
    fn pred_scalar2v2(_: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.pred.xor"]
    fn pred_xor(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.set.qfext"]
    fn set_qfext(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.shuffeqh"]
    fn shuffeqh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.shuffeqw"]
    fn shuffeqw(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.v6mpyhubs10"]
    fn v6mpyhubs10(_: HvxVectorPair, _: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.v6mpyhubs10.vxx"]
    fn v6mpyhubs10_vxx(
        _: HvxVectorPair,
        _: HvxVectorPair,
        _: HvxVectorPair,
        _: i32,
    ) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.v6mpyvubs10"]
    fn v6mpyvubs10(_: HvxVectorPair, _: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.v6mpyvubs10.vxx"]
    fn v6mpyvubs10_vxx(
        _: HvxVectorPair,
        _: HvxVectorPair,
        _: HvxVectorPair,
        _: i32,
    ) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vS32b.nqpred.ai"]
    fn vS32b_nqpred_ai(_: HvxVector, _: *mut HvxVector, _: HvxVector) -> ();
    #[link_name = "llvm.hexagon.V6.vS32b.nt.nqpred.ai"]
    fn vS32b_nt_nqpred_ai(_: HvxVector, _: *mut HvxVector, _: HvxVector) -> ();
    #[link_name = "llvm.hexagon.V6.vS32b.nt.qpred.ai"]
    fn vS32b_nt_qpred_ai(_: HvxVector, _: *mut HvxVector, _: HvxVector) -> ();
    #[link_name = "llvm.hexagon.V6.vS32b.qpred.ai"]
    fn vS32b_qpred_ai(_: HvxVector, _: *mut HvxVector, _: HvxVector) -> ();
    #[link_name = "llvm.hexagon.V6.vabs.f8"]
    fn vabs_f8(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vabs.hf"]
    fn vabs_hf(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vabs.sf"]
    fn vabs_sf(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vabsb"]
    fn vabsb(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vabsb.sat"]
    fn vabsb_sat(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vabsdiffh"]
    fn vabsdiffh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vabsdiffub"]
    fn vabsdiffub(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vabsdiffuh"]
    fn vabsdiffuh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vabsdiffw"]
    fn vabsdiffw(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vabsh"]
    fn vabsh(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vabsh.sat"]
    fn vabsh_sat(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vabsw"]
    fn vabsw(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vabsw.sat"]
    fn vabsw_sat(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vadd.hf"]
    fn vadd_hf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vadd.hf.hf"]
    fn vadd_hf_hf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vadd.qf16"]
    fn vadd_qf16(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vadd.qf16.mix"]
    fn vadd_qf16_mix(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vadd.qf32"]
    fn vadd_qf32(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vadd.qf32.mix"]
    fn vadd_qf32_mix(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vadd.sf"]
    fn vadd_sf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vadd.sf.hf"]
    fn vadd_sf_hf(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vadd.sf.sf"]
    fn vadd_sf_sf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaddb"]
    fn vaddb(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaddb.dv"]
    fn vaddb_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vaddbnq"]
    fn vaddbnq(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaddbq"]
    fn vaddbq(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaddbsat"]
    fn vaddbsat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaddbsat.dv"]
    fn vaddbsat_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vaddcarrysat"]
    fn vaddcarrysat(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaddclbh"]
    fn vaddclbh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaddclbw"]
    fn vaddclbw(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaddh"]
    fn vaddh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaddh.dv"]
    fn vaddh_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vaddhnq"]
    fn vaddhnq(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaddhq"]
    fn vaddhq(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaddhsat"]
    fn vaddhsat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaddhsat.dv"]
    fn vaddhsat_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vaddhw"]
    fn vaddhw(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vaddhw.acc"]
    fn vaddhw_acc(_: HvxVectorPair, _: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vaddubh"]
    fn vaddubh(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vaddubh.acc"]
    fn vaddubh_acc(_: HvxVectorPair, _: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vaddubsat"]
    fn vaddubsat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaddubsat.dv"]
    fn vaddubsat_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vaddububb.sat"]
    fn vaddububb_sat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vadduhsat"]
    fn vadduhsat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vadduhsat.dv"]
    fn vadduhsat_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vadduhw"]
    fn vadduhw(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vadduhw.acc"]
    fn vadduhw_acc(_: HvxVectorPair, _: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vadduwsat"]
    fn vadduwsat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vadduwsat.dv"]
    fn vadduwsat_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vaddw"]
    fn vaddw(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaddw.dv"]
    fn vaddw_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vaddwnq"]
    fn vaddwnq(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaddwq"]
    fn vaddwq(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaddwsat"]
    fn vaddwsat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaddwsat.dv"]
    fn vaddwsat_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.valignb"]
    fn valignb(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.valignbi"]
    fn valignbi(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vand"]
    fn vand(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vandnqrt"]
    fn vandnqrt(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vandnqrt.acc"]
    fn vandnqrt_acc(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vandqrt"]
    fn vandqrt(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vandqrt.acc"]
    fn vandqrt_acc(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vandvnqv"]
    fn vandvnqv(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vandvqv"]
    fn vandvqv(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vandvrt"]
    fn vandvrt(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vandvrt.acc"]
    fn vandvrt_acc(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaslh"]
    fn vaslh(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaslh.acc"]
    fn vaslh_acc(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaslhv"]
    fn vaslhv(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaslw"]
    fn vaslw(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaslw.acc"]
    fn vaslw_acc(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaslwv"]
    fn vaslwv(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vasr.into"]
    fn vasr_into(_: HvxVectorPair, _: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vasrh"]
    fn vasrh(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vasrh.acc"]
    fn vasrh_acc(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vasrhbrndsat"]
    fn vasrhbrndsat(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vasrhbsat"]
    fn vasrhbsat(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vasrhubrndsat"]
    fn vasrhubrndsat(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vasrhubsat"]
    fn vasrhubsat(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vasrhv"]
    fn vasrhv(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vasruhubrndsat"]
    fn vasruhubrndsat(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vasruhubsat"]
    fn vasruhubsat(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vasruwuhrndsat"]
    fn vasruwuhrndsat(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vasruwuhsat"]
    fn vasruwuhsat(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vasrvuhubrndsat"]
    fn vasrvuhubrndsat(_: HvxVectorPair, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vasrvuhubsat"]
    fn vasrvuhubsat(_: HvxVectorPair, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vasrvwuhrndsat"]
    fn vasrvwuhrndsat(_: HvxVectorPair, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vasrvwuhsat"]
    fn vasrvwuhsat(_: HvxVectorPair, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vasrw"]
    fn vasrw(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vasrw.acc"]
    fn vasrw_acc(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vasrwh"]
    fn vasrwh(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vasrwhrndsat"]
    fn vasrwhrndsat(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vasrwhsat"]
    fn vasrwhsat(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vasrwuhrndsat"]
    fn vasrwuhrndsat(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vasrwuhsat"]
    fn vasrwuhsat(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vasrwv"]
    fn vasrwv(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vassign"]
    fn vassign(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vassign.fp"]
    fn vassign_fp(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vassignp"]
    fn vassignp(_: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vavgb"]
    fn vavgb(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vavgbrnd"]
    fn vavgbrnd(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vavgh"]
    fn vavgh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vavghrnd"]
    fn vavghrnd(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vavgub"]
    fn vavgub(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vavgubrnd"]
    fn vavgubrnd(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vavguh"]
    fn vavguh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vavguhrnd"]
    fn vavguhrnd(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vavguw"]
    fn vavguw(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vavguwrnd"]
    fn vavguwrnd(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vavgw"]
    fn vavgw(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vavgwrnd"]
    fn vavgwrnd(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vcl0h"]
    fn vcl0h(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vcl0w"]
    fn vcl0w(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vcombine"]
    fn vcombine(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vconv.h.hf"]
    fn vconv_h_hf(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vconv.hf.h"]
    fn vconv_hf_h(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vconv.hf.qf16"]
    fn vconv_hf_qf16(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vconv.hf.qf32"]
    fn vconv_hf_qf32(_: HvxVectorPair) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vconv.sf.qf32"]
    fn vconv_sf_qf32(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vconv.sf.w"]
    fn vconv_sf_w(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vconv.w.sf"]
    fn vconv_w_sf(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vcvt2.hf.b"]
    fn vcvt2_hf_b(_: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vcvt2.hf.ub"]
    fn vcvt2_hf_ub(_: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vcvt.b.hf"]
    fn vcvt_b_hf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vcvt.h.hf"]
    fn vcvt_h_hf(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vcvt.hf.b"]
    fn vcvt_hf_b(_: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vcvt.hf.f8"]
    fn vcvt_hf_f8(_: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vcvt.hf.h"]
    fn vcvt_hf_h(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vcvt.hf.sf"]
    fn vcvt_hf_sf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vcvt.hf.ub"]
    fn vcvt_hf_ub(_: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vcvt.hf.uh"]
    fn vcvt_hf_uh(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vcvt.sf.hf"]
    fn vcvt_sf_hf(_: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vcvt.ub.hf"]
    fn vcvt_ub_hf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vcvt.uh.hf"]
    fn vcvt_uh_hf(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vd0"]
    fn vd0() -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdd0"]
    fn vdd0() -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vdealb"]
    fn vdealb(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdealb4w"]
    fn vdealb4w(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdealh"]
    fn vdealh(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdealvdd"]
    fn vdealvdd(_: HvxVector, _: HvxVector, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vdelta"]
    fn vdelta(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdmpy.sf.hf"]
    fn vdmpy_sf_hf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdmpy.sf.hf.acc"]
    fn vdmpy_sf_hf_acc(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdmpybus"]
    fn vdmpybus(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdmpybus.acc"]
    fn vdmpybus_acc(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdmpybus.dv"]
    fn vdmpybus_dv(_: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vdmpybus.dv.acc"]
    fn vdmpybus_dv_acc(_: HvxVectorPair, _: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vdmpyhb"]
    fn vdmpyhb(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdmpyhb.acc"]
    fn vdmpyhb_acc(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdmpyhb.dv"]
    fn vdmpyhb_dv(_: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vdmpyhb.dv.acc"]
    fn vdmpyhb_dv_acc(_: HvxVectorPair, _: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vdmpyhisat"]
    fn vdmpyhisat(_: HvxVectorPair, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdmpyhisat.acc"]
    fn vdmpyhisat_acc(_: HvxVector, _: HvxVectorPair, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdmpyhsat"]
    fn vdmpyhsat(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdmpyhsat.acc"]
    fn vdmpyhsat_acc(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdmpyhsuisat"]
    fn vdmpyhsuisat(_: HvxVectorPair, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdmpyhsuisat.acc"]
    fn vdmpyhsuisat_acc(_: HvxVector, _: HvxVectorPair, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdmpyhsusat"]
    fn vdmpyhsusat(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdmpyhsusat.acc"]
    fn vdmpyhsusat_acc(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdmpyhvsat"]
    fn vdmpyhvsat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdmpyhvsat.acc"]
    fn vdmpyhvsat_acc(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdsaduh"]
    fn vdsaduh(_: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vdsaduh.acc"]
    fn vdsaduh_acc(_: HvxVectorPair, _: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.veqb"]
    fn veqb(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.veqb.and"]
    fn veqb_and(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.veqb.or"]
    fn veqb_or(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.veqb.xor"]
    fn veqb_xor(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.veqh"]
    fn veqh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.veqh.and"]
    fn veqh_and(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.veqh.or"]
    fn veqh_or(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.veqh.xor"]
    fn veqh_xor(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.veqw"]
    fn veqw(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.veqw.and"]
    fn veqw_and(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.veqw.or"]
    fn veqw_or(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.veqw.xor"]
    fn veqw_xor(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vfmax.f8"]
    fn vfmax_f8(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vfmax.hf"]
    fn vfmax_hf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vfmax.sf"]
    fn vfmax_sf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vfmin.f8"]
    fn vfmin_f8(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vfmin.hf"]
    fn vfmin_hf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vfmin.sf"]
    fn vfmin_sf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vfneg.f8"]
    fn vfneg_f8(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vfneg.hf"]
    fn vfneg_hf(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vfneg.sf"]
    fn vfneg_sf(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgathermh"]
    fn vgathermh(_: *mut HvxVector, _: i32, _: i32, _: HvxVector) -> ();
    #[link_name = "llvm.hexagon.V6.vgathermhq"]
    fn vgathermhq(_: *mut HvxVector, _: HvxVector, _: i32, _: i32, _: HvxVector) -> ();
    #[link_name = "llvm.hexagon.V6.vgathermhw"]
    fn vgathermhw(_: *mut HvxVector, _: i32, _: i32, _: HvxVectorPair) -> ();
    #[link_name = "llvm.hexagon.V6.vgathermhwq"]
    fn vgathermhwq(_: *mut HvxVector, _: HvxVector, _: i32, _: i32, _: HvxVectorPair) -> ();
    #[link_name = "llvm.hexagon.V6.vgathermw"]
    fn vgathermw(_: *mut HvxVector, _: i32, _: i32, _: HvxVector) -> ();
    #[link_name = "llvm.hexagon.V6.vgathermwq"]
    fn vgathermwq(_: *mut HvxVector, _: HvxVector, _: i32, _: i32, _: HvxVector) -> ();
    #[link_name = "llvm.hexagon.V6.vgtb"]
    fn vgtb(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgtb.and"]
    fn vgtb_and(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgtb.or"]
    fn vgtb_or(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgtb.xor"]
    fn vgtb_xor(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgth"]
    fn vgth(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgth.and"]
    fn vgth_and(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgth.or"]
    fn vgth_or(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgth.xor"]
    fn vgth_xor(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgthf"]
    fn vgthf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgthf.and"]
    fn vgthf_and(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgthf.or"]
    fn vgthf_or(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgthf.xor"]
    fn vgthf_xor(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgtsf"]
    fn vgtsf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgtsf.and"]
    fn vgtsf_and(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgtsf.or"]
    fn vgtsf_or(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgtsf.xor"]
    fn vgtsf_xor(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgtub"]
    fn vgtub(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgtub.and"]
    fn vgtub_and(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgtub.or"]
    fn vgtub_or(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgtub.xor"]
    fn vgtub_xor(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgtuh"]
    fn vgtuh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgtuh.and"]
    fn vgtuh_and(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgtuh.or"]
    fn vgtuh_or(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgtuh.xor"]
    fn vgtuh_xor(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgtuw"]
    fn vgtuw(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgtuw.and"]
    fn vgtuw_and(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgtuw.or"]
    fn vgtuw_or(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgtuw.xor"]
    fn vgtuw_xor(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgtw"]
    fn vgtw(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgtw.and"]
    fn vgtw_and(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgtw.or"]
    fn vgtw_or(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgtw.xor"]
    fn vgtw_xor(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vinsertwr"]
    fn vinsertwr(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vlalignb"]
    fn vlalignb(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vlalignbi"]
    fn vlalignbi(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vlsrb"]
    fn vlsrb(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vlsrh"]
    fn vlsrh(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vlsrhv"]
    fn vlsrhv(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vlsrw"]
    fn vlsrw(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vlsrwv"]
    fn vlsrwv(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vlutvvb"]
    fn vlutvvb(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vlutvvb.nm"]
    fn vlutvvb_nm(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vlutvvb.oracc"]
    fn vlutvvb_oracc(_: HvxVector, _: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vlutvvb.oracci"]
    fn vlutvvb_oracci(_: HvxVector, _: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vlutvvbi"]
    fn vlutvvbi(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vlutvwh"]
    fn vlutvwh(_: HvxVector, _: HvxVector, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vlutvwh.nm"]
    fn vlutvwh_nm(_: HvxVector, _: HvxVector, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vlutvwh.oracc"]
    fn vlutvwh_oracc(_: HvxVectorPair, _: HvxVector, _: HvxVector, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vlutvwh.oracci"]
    fn vlutvwh_oracci(_: HvxVectorPair, _: HvxVector, _: HvxVector, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vlutvwhi"]
    fn vlutvwhi(_: HvxVector, _: HvxVector, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmax.hf"]
    fn vmax_hf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmax.sf"]
    fn vmax_sf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmaxb"]
    fn vmaxb(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmaxh"]
    fn vmaxh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmaxub"]
    fn vmaxub(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmaxuh"]
    fn vmaxuh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmaxw"]
    fn vmaxw(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmin.hf"]
    fn vmin_hf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmin.sf"]
    fn vmin_sf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vminb"]
    fn vminb(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vminh"]
    fn vminh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vminub"]
    fn vminub(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vminuh"]
    fn vminuh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vminw"]
    fn vminw(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpabus"]
    fn vmpabus(_: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpabus.acc"]
    fn vmpabus_acc(_: HvxVectorPair, _: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpabusv"]
    fn vmpabusv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpabuu"]
    fn vmpabuu(_: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpabuu.acc"]
    fn vmpabuu_acc(_: HvxVectorPair, _: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpabuuv"]
    fn vmpabuuv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpahb"]
    fn vmpahb(_: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpahb.acc"]
    fn vmpahb_acc(_: HvxVectorPair, _: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpauhb"]
    fn vmpauhb(_: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpauhb.acc"]
    fn vmpauhb_acc(_: HvxVectorPair, _: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpy.hf.hf"]
    fn vmpy_hf_hf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpy.hf.hf.acc"]
    fn vmpy_hf_hf_acc(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpy.qf16"]
    fn vmpy_qf16(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpy.qf16.hf"]
    fn vmpy_qf16_hf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpy.qf16.mix.hf"]
    fn vmpy_qf16_mix_hf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpy.qf32"]
    fn vmpy_qf32(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpy.qf32.hf"]
    fn vmpy_qf32_hf(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpy.qf32.mix.hf"]
    fn vmpy_qf32_mix_hf(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpy.qf32.qf16"]
    fn vmpy_qf32_qf16(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpy.qf32.sf"]
    fn vmpy_qf32_sf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpy.sf.hf"]
    fn vmpy_sf_hf(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpy.sf.hf.acc"]
    fn vmpy_sf_hf_acc(_: HvxVectorPair, _: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpy.sf.sf"]
    fn vmpy_sf_sf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpybus"]
    fn vmpybus(_: HvxVector, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpybus.acc"]
    fn vmpybus_acc(_: HvxVectorPair, _: HvxVector, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpybusv"]
    fn vmpybusv(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpybusv.acc"]
    fn vmpybusv_acc(_: HvxVectorPair, _: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpybv"]
    fn vmpybv(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpybv.acc"]
    fn vmpybv_acc(_: HvxVectorPair, _: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyewuh"]
    fn vmpyewuh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyewuh.64"]
    fn vmpyewuh_64(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyh"]
    fn vmpyh(_: HvxVector, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyh.acc"]
    fn vmpyh_acc(_: HvxVectorPair, _: HvxVector, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyhsat.acc"]
    fn vmpyhsat_acc(_: HvxVectorPair, _: HvxVector, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyhsrs"]
    fn vmpyhsrs(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyhss"]
    fn vmpyhss(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyhus"]
    fn vmpyhus(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyhus.acc"]
    fn vmpyhus_acc(_: HvxVectorPair, _: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyhv"]
    fn vmpyhv(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyhv.acc"]
    fn vmpyhv_acc(_: HvxVectorPair, _: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyhvsrs"]
    fn vmpyhvsrs(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyieoh"]
    fn vmpyieoh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyiewh.acc"]
    fn vmpyiewh_acc(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyiewuh"]
    fn vmpyiewuh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyiewuh.acc"]
    fn vmpyiewuh_acc(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyih"]
    fn vmpyih(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyih.acc"]
    fn vmpyih_acc(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyihb"]
    fn vmpyihb(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyihb.acc"]
    fn vmpyihb_acc(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyiowh"]
    fn vmpyiowh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyiwb"]
    fn vmpyiwb(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyiwb.acc"]
    fn vmpyiwb_acc(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyiwh"]
    fn vmpyiwh(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyiwh.acc"]
    fn vmpyiwh_acc(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyiwub"]
    fn vmpyiwub(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyiwub.acc"]
    fn vmpyiwub_acc(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyowh"]
    fn vmpyowh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyowh.64.acc"]
    fn vmpyowh_64_acc(_: HvxVectorPair, _: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyowh.rnd"]
    fn vmpyowh_rnd(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyowh.rnd.sacc"]
    fn vmpyowh_rnd_sacc(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyowh.sacc"]
    fn vmpyowh_sacc(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyub"]
    fn vmpyub(_: HvxVector, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyub.acc"]
    fn vmpyub_acc(_: HvxVectorPair, _: HvxVector, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyubv"]
    fn vmpyubv(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyubv.acc"]
    fn vmpyubv_acc(_: HvxVectorPair, _: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyuh"]
    fn vmpyuh(_: HvxVector, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyuh.acc"]
    fn vmpyuh_acc(_: HvxVectorPair, _: HvxVector, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyuhe"]
    fn vmpyuhe(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyuhe.acc"]
    fn vmpyuhe_acc(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyuhv"]
    fn vmpyuhv(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyuhv.acc"]
    fn vmpyuhv_acc(_: HvxVectorPair, _: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyuhvs"]
    fn vmpyuhvs(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmux"]
    fn vmux(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vnavgb"]
    fn vnavgb(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vnavgh"]
    fn vnavgh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vnavgub"]
    fn vnavgub(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vnavgw"]
    fn vnavgw(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vnormamth"]
    fn vnormamth(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vnormamtw"]
    fn vnormamtw(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vnot"]
    fn vnot(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vor"]
    fn vor(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vpackeb"]
    fn vpackeb(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vpackeh"]
    fn vpackeh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vpackhb.sat"]
    fn vpackhb_sat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vpackhub.sat"]
    fn vpackhub_sat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vpackob"]
    fn vpackob(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vpackoh"]
    fn vpackoh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vpackwh.sat"]
    fn vpackwh_sat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vpackwuh.sat"]
    fn vpackwuh_sat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vpopcounth"]
    fn vpopcounth(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vprefixqb"]
    fn vprefixqb(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vprefixqh"]
    fn vprefixqh(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vprefixqw"]
    fn vprefixqw(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vrdelta"]
    fn vrdelta(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vrmpybus"]
    fn vrmpybus(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vrmpybus.acc"]
    fn vrmpybus_acc(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vrmpybusi"]
    fn vrmpybusi(_: HvxVectorPair, _: i32, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vrmpybusi.acc"]
    fn vrmpybusi_acc(_: HvxVectorPair, _: HvxVectorPair, _: i32, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vrmpybusv"]
    fn vrmpybusv(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vrmpybusv.acc"]
    fn vrmpybusv_acc(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vrmpybv"]
    fn vrmpybv(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vrmpybv.acc"]
    fn vrmpybv_acc(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vrmpyub"]
    fn vrmpyub(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vrmpyub.acc"]
    fn vrmpyub_acc(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vrmpyubi"]
    fn vrmpyubi(_: HvxVectorPair, _: i32, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vrmpyubi.acc"]
    fn vrmpyubi_acc(_: HvxVectorPair, _: HvxVectorPair, _: i32, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vrmpyubv"]
    fn vrmpyubv(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vrmpyubv.acc"]
    fn vrmpyubv_acc(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vror"]
    fn vror(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vrotr"]
    fn vrotr(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vroundhb"]
    fn vroundhb(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vroundhub"]
    fn vroundhub(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vrounduhub"]
    fn vrounduhub(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vrounduwuh"]
    fn vrounduwuh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vroundwh"]
    fn vroundwh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vroundwuh"]
    fn vroundwuh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vrsadubi"]
    fn vrsadubi(_: HvxVectorPair, _: i32, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vrsadubi.acc"]
    fn vrsadubi_acc(_: HvxVectorPair, _: HvxVectorPair, _: i32, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vsatdw"]
    fn vsatdw(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsathub"]
    fn vsathub(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsatuwuh"]
    fn vsatuwuh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsatwh"]
    fn vsatwh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsb"]
    fn vsb(_: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vscattermh"]
    fn vscattermh(_: i32, _: i32, _: HvxVector, _: HvxVector) -> ();
    #[link_name = "llvm.hexagon.V6.vscattermh.add"]
    fn vscattermh_add(_: i32, _: i32, _: HvxVector, _: HvxVector) -> ();
    #[link_name = "llvm.hexagon.V6.vscattermhq"]
    fn vscattermhq(_: HvxVector, _: i32, _: i32, _: HvxVector, _: HvxVector) -> ();
    #[link_name = "llvm.hexagon.V6.vscattermhw"]
    fn vscattermhw(_: i32, _: i32, _: HvxVectorPair, _: HvxVector) -> ();
    #[link_name = "llvm.hexagon.V6.vscattermhw.add"]
    fn vscattermhw_add(_: i32, _: i32, _: HvxVectorPair, _: HvxVector) -> ();
    #[link_name = "llvm.hexagon.V6.vscattermhwq"]
    fn vscattermhwq(_: HvxVector, _: i32, _: i32, _: HvxVectorPair, _: HvxVector) -> ();
    #[link_name = "llvm.hexagon.V6.vscattermw"]
    fn vscattermw(_: i32, _: i32, _: HvxVector, _: HvxVector) -> ();
    #[link_name = "llvm.hexagon.V6.vscattermw.add"]
    fn vscattermw_add(_: i32, _: i32, _: HvxVector, _: HvxVector) -> ();
    #[link_name = "llvm.hexagon.V6.vscattermwq"]
    fn vscattermwq(_: HvxVector, _: i32, _: i32, _: HvxVector, _: HvxVector) -> ();
    #[link_name = "llvm.hexagon.V6.vsh"]
    fn vsh(_: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vshufeh"]
    fn vshufeh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vshuffb"]
    fn vshuffb(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vshuffeb"]
    fn vshuffeb(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vshuffh"]
    fn vshuffh(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vshuffob"]
    fn vshuffob(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vshuffvdd"]
    fn vshuffvdd(_: HvxVector, _: HvxVector, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vshufoeb"]
    fn vshufoeb(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vshufoeh"]
    fn vshufoeh(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vshufoh"]
    fn vshufoh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsub.hf"]
    fn vsub_hf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsub.hf.hf"]
    fn vsub_hf_hf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsub.qf16"]
    fn vsub_qf16(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsub.qf16.mix"]
    fn vsub_qf16_mix(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsub.qf32"]
    fn vsub_qf32(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsub.qf32.mix"]
    fn vsub_qf32_mix(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsub.sf"]
    fn vsub_sf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsub.sf.hf"]
    fn vsub_sf_hf(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vsub.sf.sf"]
    fn vsub_sf_sf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsubb"]
    fn vsubb(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsubb.dv"]
    fn vsubb_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vsubbnq"]
    fn vsubbnq(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsubbq"]
    fn vsubbq(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsubbsat"]
    fn vsubbsat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsubbsat.dv"]
    fn vsubbsat_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vsubh"]
    fn vsubh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsubh.dv"]
    fn vsubh_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vsubhnq"]
    fn vsubhnq(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsubhq"]
    fn vsubhq(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsubhsat"]
    fn vsubhsat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsubhsat.dv"]
    fn vsubhsat_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vsubhw"]
    fn vsubhw(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vsububh"]
    fn vsububh(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vsububsat"]
    fn vsububsat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsububsat.dv"]
    fn vsububsat_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vsubububb.sat"]
    fn vsubububb_sat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsubuhsat"]
    fn vsubuhsat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsubuhsat.dv"]
    fn vsubuhsat_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vsubuhw"]
    fn vsubuhw(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vsubuwsat"]
    fn vsubuwsat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsubuwsat.dv"]
    fn vsubuwsat_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vsubw"]
    fn vsubw(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsubw.dv"]
    fn vsubw_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vsubwnq"]
    fn vsubwnq(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsubwq"]
    fn vsubwq(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsubwsat"]
    fn vsubwsat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsubwsat.dv"]
    fn vsubwsat_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vswap"]
    fn vswap(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vtmpyb"]
    fn vtmpyb(_: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vtmpyb.acc"]
    fn vtmpyb_acc(_: HvxVectorPair, _: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vtmpybus"]
    fn vtmpybus(_: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vtmpybus.acc"]
    fn vtmpybus_acc(_: HvxVectorPair, _: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vtmpyhb"]
    fn vtmpyhb(_: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vtmpyhb.acc"]
    fn vtmpyhb_acc(_: HvxVectorPair, _: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vunpackb"]
    fn vunpackb(_: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vunpackh"]
    fn vunpackh(_: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vunpackob"]
    fn vunpackob(_: HvxVectorPair, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vunpackoh"]
    fn vunpackoh(_: HvxVectorPair, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vunpackub"]
    fn vunpackub(_: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vunpackuh"]
    fn vunpackuh(_: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vxor"]
    fn vxor(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vzb"]
    fn vzb(_: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vzh"]
    fn vzh(_: HvxVector) -> HvxVectorPair;
}

/// `Rd32=vextract(Vu32,Rs32)`
///
/// Instruction Type: LD
/// Execution Slots: SLOT0
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(extractw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_vextract_VR(vu: HvxVector, rs: i32) -> i32 {
    extractw(vu, rs)
}

/// `Vd32=hi(Vss32)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(hi))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_V_hi_W(vss: HvxVectorPair) -> HvxVector {
    hi(vss)
}

/// `Vd32=lo(Vss32)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(lo))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_V_lo_W(vss: HvxVectorPair) -> HvxVector {
    lo(vss)
}

/// `Vd32=vsplat(Rt32)`
///
/// Instruction Type: CVI_VX_LATE
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(lvsplatw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_V_vsplat_R(rt: i32) -> HvxVector {
    lvsplatw(rt)
}

/// `Vd32.uh=vabsdiff(Vu32.h,Vv32.h)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vabsdiffh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vuh_vabsdiff_VhVh(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vabsdiffh(vu, vv)
}

/// `Vd32.ub=vabsdiff(Vu32.ub,Vv32.ub)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vabsdiffub))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vub_vabsdiff_VubVub(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vabsdiffub(vu, vv)
}

/// `Vd32.uh=vabsdiff(Vu32.uh,Vv32.uh)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vabsdiffuh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vuh_vabsdiff_VuhVuh(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vabsdiffuh(vu, vv)
}

/// `Vd32.uw=vabsdiff(Vu32.w,Vv32.w)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vabsdiffw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vuw_vabsdiff_VwVw(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vabsdiffw(vu, vv)
}

/// `Vd32.h=vabs(Vu32.h)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vabsh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vh_vabs_Vh(vu: HvxVector) -> HvxVector {
    vabsh(vu)
}

/// `Vd32.h=vabs(Vu32.h):sat`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vabsh_sat))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vh_vabs_Vh_sat(vu: HvxVector) -> HvxVector {
    vabsh_sat(vu)
}

/// `Vd32.w=vabs(Vu32.w)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vabsw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vabs_Vw(vu: HvxVector) -> HvxVector {
    vabsw(vu)
}

/// `Vd32.w=vabs(Vu32.w):sat`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vabsw_sat))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vabs_Vw_sat(vu: HvxVector) -> HvxVector {
    vabsw_sat(vu)
}

/// `Vd32.b=vadd(Vu32.b,Vv32.b)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vaddb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vb_vadd_VbVb(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vaddb(vu, vv)
}

/// `Vdd32.b=vadd(Vuu32.b,Vvv32.b)`
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vaddb_dv))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wb_vadd_WbWb(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
    vaddb_dv(vuu, vvv)
}

/// `Vd32.h=vadd(Vu32.h,Vv32.h)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vaddh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vh_vadd_VhVh(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vaddh(vu, vv)
}

/// `Vdd32.h=vadd(Vuu32.h,Vvv32.h)`
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vaddh_dv))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wh_vadd_WhWh(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
    vaddh_dv(vuu, vvv)
}

/// `Vd32.h=vadd(Vu32.h,Vv32.h):sat`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vaddhsat))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vh_vadd_VhVh_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vaddhsat(vu, vv)
}

/// `Vdd32.h=vadd(Vuu32.h,Vvv32.h):sat`
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vaddhsat_dv))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wh_vadd_WhWh_sat(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
    vaddhsat_dv(vuu, vvv)
}

/// `Vdd32.w=vadd(Vu32.h,Vv32.h)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vaddhw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Ww_vadd_VhVh(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
    vaddhw(vu, vv)
}

/// `Vdd32.h=vadd(Vu32.ub,Vv32.ub)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vaddubh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wh_vadd_VubVub(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
    vaddubh(vu, vv)
}

/// `Vd32.ub=vadd(Vu32.ub,Vv32.ub):sat`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vaddubsat))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vub_vadd_VubVub_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vaddubsat(vu, vv)
}

/// `Vdd32.ub=vadd(Vuu32.ub,Vvv32.ub):sat`
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vaddubsat_dv))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wub_vadd_WubWub_sat(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
    vaddubsat_dv(vuu, vvv)
}

/// `Vd32.uh=vadd(Vu32.uh,Vv32.uh):sat`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vadduhsat))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vuh_vadd_VuhVuh_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vadduhsat(vu, vv)
}

/// `Vdd32.uh=vadd(Vuu32.uh,Vvv32.uh):sat`
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vadduhsat_dv))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wuh_vadd_WuhWuh_sat(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
    vadduhsat_dv(vuu, vvv)
}

/// `Vdd32.w=vadd(Vu32.uh,Vv32.uh)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vadduhw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Ww_vadd_VuhVuh(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
    vadduhw(vu, vv)
}

/// `Vd32.w=vadd(Vu32.w,Vv32.w)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vaddw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vadd_VwVw(vu: HvxVector, vv: HvxVector) -> HvxVector {
    simd_add(vu, vv)
}

/// `Vdd32.w=vadd(Vuu32.w,Vvv32.w)`
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vaddw_dv))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Ww_vadd_WwWw(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
    vaddw_dv(vuu, vvv)
}

/// `Vd32.w=vadd(Vu32.w,Vv32.w):sat`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vaddwsat))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vadd_VwVw_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vaddwsat(vu, vv)
}

/// `Vdd32.w=vadd(Vuu32.w,Vvv32.w):sat`
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vaddwsat_dv))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Ww_vadd_WwWw_sat(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
    vaddwsat_dv(vuu, vvv)
}

/// `Vd32=valign(Vu32,Vv32,Rt8)`
///
/// Instruction Type: CVI_VP
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(valignb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_V_valign_VVR(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVector {
    valignb(vu, vv, rt)
}

/// `Vd32=valign(Vu32,Vv32,#u3)`
///
/// Instruction Type: CVI_VP
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(valignbi))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_V_valign_VVI(vu: HvxVector, vv: HvxVector, iu3: i32) -> HvxVector {
    valignbi(vu, vv, iu3)
}

/// `Vd32=vand(Vu32,Vv32)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vand))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_V_vand_VV(vu: HvxVector, vv: HvxVector) -> HvxVector {
    simd_and(vu, vv)
}

/// `Vd32.h=vasl(Vu32.h,Rt32)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vaslh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vh_vasl_VhR(vu: HvxVector, rt: i32) -> HvxVector {
    vaslh(vu, rt)
}

/// `Vd32.h=vasl(Vu32.h,Vv32.h)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vaslhv))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vh_vasl_VhVh(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vaslhv(vu, vv)
}

/// `Vd32.w=vasl(Vu32.w,Rt32)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vaslw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vasl_VwR(vu: HvxVector, rt: i32) -> HvxVector {
    vaslw(vu, rt)
}

/// `Vx32.w+=vasl(Vu32.w,Rt32)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vaslw_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vaslacc_VwVwR(vx: HvxVector, vu: HvxVector, rt: i32) -> HvxVector {
    vaslw_acc(vx, vu, rt)
}

/// `Vd32.w=vasl(Vu32.w,Vv32.w)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vaslwv))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vasl_VwVw(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vaslwv(vu, vv)
}

/// `Vd32.h=vasr(Vu32.h,Rt32)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vasrh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vh_vasr_VhR(vu: HvxVector, rt: i32) -> HvxVector {
    vasrh(vu, rt)
}

/// `Vd32.b=vasr(Vu32.h,Vv32.h,Rt8):rnd:sat`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vasrhbrndsat))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vb_vasr_VhVhR_rnd_sat(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVector {
    vasrhbrndsat(vu, vv, rt)
}

/// `Vd32.ub=vasr(Vu32.h,Vv32.h,Rt8):rnd:sat`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vasrhubrndsat))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vub_vasr_VhVhR_rnd_sat(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVector {
    vasrhubrndsat(vu, vv, rt)
}

/// `Vd32.ub=vasr(Vu32.h,Vv32.h,Rt8):sat`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vasrhubsat))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vub_vasr_VhVhR_sat(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVector {
    vasrhubsat(vu, vv, rt)
}

/// `Vd32.h=vasr(Vu32.h,Vv32.h)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vasrhv))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vh_vasr_VhVh(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vasrhv(vu, vv)
}

/// `Vd32.w=vasr(Vu32.w,Rt32)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vasrw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vasr_VwR(vu: HvxVector, rt: i32) -> HvxVector {
    vasrw(vu, rt)
}

/// `Vx32.w+=vasr(Vu32.w,Rt32)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vasrw_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vasracc_VwVwR(vx: HvxVector, vu: HvxVector, rt: i32) -> HvxVector {
    vasrw_acc(vx, vu, rt)
}

/// `Vd32.h=vasr(Vu32.w,Vv32.w,Rt8)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vasrwh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vh_vasr_VwVwR(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVector {
    vasrwh(vu, vv, rt)
}

/// `Vd32.h=vasr(Vu32.w,Vv32.w,Rt8):rnd:sat`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vasrwhrndsat))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vh_vasr_VwVwR_rnd_sat(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVector {
    vasrwhrndsat(vu, vv, rt)
}

/// `Vd32.h=vasr(Vu32.w,Vv32.w,Rt8):sat`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vasrwhsat))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vh_vasr_VwVwR_sat(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVector {
    vasrwhsat(vu, vv, rt)
}

/// `Vd32.uh=vasr(Vu32.w,Vv32.w,Rt8):sat`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vasrwuhsat))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vuh_vasr_VwVwR_sat(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVector {
    vasrwuhsat(vu, vv, rt)
}

/// `Vd32.w=vasr(Vu32.w,Vv32.w)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vasrwv))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vasr_VwVw(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vasrwv(vu, vv)
}

/// `Vd32=Vu32`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vassign))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_V_equals_V(vu: HvxVector) -> HvxVector {
    vassign(vu)
}

/// `Vdd32=Vuu32`
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vassignp))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_W_equals_W(vuu: HvxVectorPair) -> HvxVectorPair {
    vassignp(vuu)
}

/// `Vd32.h=vavg(Vu32.h,Vv32.h)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vavgh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vh_vavg_VhVh(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vavgh(vu, vv)
}

/// `Vd32.h=vavg(Vu32.h,Vv32.h):rnd`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vavghrnd))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vh_vavg_VhVh_rnd(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vavghrnd(vu, vv)
}

/// `Vd32.ub=vavg(Vu32.ub,Vv32.ub)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vavgub))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vub_vavg_VubVub(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vavgub(vu, vv)
}

/// `Vd32.ub=vavg(Vu32.ub,Vv32.ub):rnd`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vavgubrnd))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vub_vavg_VubVub_rnd(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vavgubrnd(vu, vv)
}

/// `Vd32.uh=vavg(Vu32.uh,Vv32.uh)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vavguh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vuh_vavg_VuhVuh(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vavguh(vu, vv)
}

/// `Vd32.uh=vavg(Vu32.uh,Vv32.uh):rnd`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vavguhrnd))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vuh_vavg_VuhVuh_rnd(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vavguhrnd(vu, vv)
}

/// `Vd32.w=vavg(Vu32.w,Vv32.w)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vavgw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vavg_VwVw(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vavgw(vu, vv)
}

/// `Vd32.w=vavg(Vu32.w,Vv32.w):rnd`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vavgwrnd))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vavg_VwVw_rnd(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vavgwrnd(vu, vv)
}

/// `Vd32.uh=vcl0(Vu32.uh)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vcl0h))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vuh_vcl0_Vuh(vu: HvxVector) -> HvxVector {
    vcl0h(vu)
}

/// `Vd32.uw=vcl0(Vu32.uw)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vcl0w))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vuw_vcl0_Vuw(vu: HvxVector) -> HvxVector {
    vcl0w(vu)
}

/// `Vdd32=vcombine(Vu32,Vv32)`
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vcombine))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_W_vcombine_VV(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
    vcombine(vu, vv)
}

/// `Vd32=#0`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vd0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_V_vzero() -> HvxVector {
    vd0()
}

/// `Vd32.b=vdeal(Vu32.b)`
///
/// Instruction Type: CVI_VP
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vdealb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vb_vdeal_Vb(vu: HvxVector) -> HvxVector {
    vdealb(vu)
}

/// `Vd32.b=vdeale(Vu32.b,Vv32.b)`
///
/// Instruction Type: CVI_VP
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vdealb4w))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vb_vdeale_VbVb(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vdealb4w(vu, vv)
}

/// `Vd32.h=vdeal(Vu32.h)`
///
/// Instruction Type: CVI_VP
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vdealh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vh_vdeal_Vh(vu: HvxVector) -> HvxVector {
    vdealh(vu)
}

/// `Vdd32=vdeal(Vu32,Vv32,Rt8)`
///
/// Instruction Type: CVI_VP_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vdealvdd))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_W_vdeal_VVR(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVectorPair {
    vdealvdd(vu, vv, rt)
}

/// `Vd32=vdelta(Vu32,Vv32)`
///
/// Instruction Type: CVI_VP
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vdelta))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_V_vdelta_VV(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vdelta(vu, vv)
}

/// `Vd32.h=vdmpy(Vu32.ub,Rt32.b)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vdmpybus))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vh_vdmpy_VubRb(vu: HvxVector, rt: i32) -> HvxVector {
    vdmpybus(vu, rt)
}

/// `Vx32.h+=vdmpy(Vu32.ub,Rt32.b)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vdmpybus_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vh_vdmpyacc_VhVubRb(vx: HvxVector, vu: HvxVector, rt: i32) -> HvxVector {
    vdmpybus_acc(vx, vu, rt)
}

/// `Vdd32.h=vdmpy(Vuu32.ub,Rt32.b)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vdmpybus_dv))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wh_vdmpy_WubRb(vuu: HvxVectorPair, rt: i32) -> HvxVectorPair {
    vdmpybus_dv(vuu, rt)
}

/// `Vxx32.h+=vdmpy(Vuu32.ub,Rt32.b)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vdmpybus_dv_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wh_vdmpyacc_WhWubRb(
    vxx: HvxVectorPair,
    vuu: HvxVectorPair,
    rt: i32,
) -> HvxVectorPair {
    vdmpybus_dv_acc(vxx, vuu, rt)
}

/// `Vd32.w=vdmpy(Vu32.h,Rt32.b)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vdmpyhb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vdmpy_VhRb(vu: HvxVector, rt: i32) -> HvxVector {
    vdmpyhb(vu, rt)
}

/// `Vx32.w+=vdmpy(Vu32.h,Rt32.b)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vdmpyhb_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vdmpyacc_VwVhRb(vx: HvxVector, vu: HvxVector, rt: i32) -> HvxVector {
    vdmpyhb_acc(vx, vu, rt)
}

/// `Vdd32.w=vdmpy(Vuu32.h,Rt32.b)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vdmpyhb_dv))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Ww_vdmpy_WhRb(vuu: HvxVectorPair, rt: i32) -> HvxVectorPair {
    vdmpyhb_dv(vuu, rt)
}

/// `Vxx32.w+=vdmpy(Vuu32.h,Rt32.b)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vdmpyhb_dv_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Ww_vdmpyacc_WwWhRb(
    vxx: HvxVectorPair,
    vuu: HvxVectorPair,
    rt: i32,
) -> HvxVectorPair {
    vdmpyhb_dv_acc(vxx, vuu, rt)
}

/// `Vd32.w=vdmpy(Vuu32.h,Rt32.h):sat`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vdmpyhisat))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vdmpy_WhRh_sat(vuu: HvxVectorPair, rt: i32) -> HvxVector {
    vdmpyhisat(vuu, rt)
}

/// `Vx32.w+=vdmpy(Vuu32.h,Rt32.h):sat`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vdmpyhisat_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vdmpyacc_VwWhRh_sat(vx: HvxVector, vuu: HvxVectorPair, rt: i32) -> HvxVector {
    vdmpyhisat_acc(vx, vuu, rt)
}

/// `Vd32.w=vdmpy(Vu32.h,Rt32.h):sat`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vdmpyhsat))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vdmpy_VhRh_sat(vu: HvxVector, rt: i32) -> HvxVector {
    vdmpyhsat(vu, rt)
}

/// `Vx32.w+=vdmpy(Vu32.h,Rt32.h):sat`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vdmpyhsat_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vdmpyacc_VwVhRh_sat(vx: HvxVector, vu: HvxVector, rt: i32) -> HvxVector {
    vdmpyhsat_acc(vx, vu, rt)
}

/// `Vd32.w=vdmpy(Vuu32.h,Rt32.uh,#1):sat`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vdmpyhsuisat))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vdmpy_WhRuh_sat(vuu: HvxVectorPair, rt: i32) -> HvxVector {
    vdmpyhsuisat(vuu, rt)
}

/// `Vx32.w+=vdmpy(Vuu32.h,Rt32.uh,#1):sat`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vdmpyhsuisat_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vdmpyacc_VwWhRuh_sat(vx: HvxVector, vuu: HvxVectorPair, rt: i32) -> HvxVector {
    vdmpyhsuisat_acc(vx, vuu, rt)
}

/// `Vd32.w=vdmpy(Vu32.h,Rt32.uh):sat`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vdmpyhsusat))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vdmpy_VhRuh_sat(vu: HvxVector, rt: i32) -> HvxVector {
    vdmpyhsusat(vu, rt)
}

/// `Vx32.w+=vdmpy(Vu32.h,Rt32.uh):sat`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vdmpyhsusat_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vdmpyacc_VwVhRuh_sat(vx: HvxVector, vu: HvxVector, rt: i32) -> HvxVector {
    vdmpyhsusat_acc(vx, vu, rt)
}

/// `Vd32.w=vdmpy(Vu32.h,Vv32.h):sat`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vdmpyhvsat))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vdmpy_VhVh_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vdmpyhvsat(vu, vv)
}

/// `Vx32.w+=vdmpy(Vu32.h,Vv32.h):sat`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vdmpyhvsat_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vdmpyacc_VwVhVh_sat(vx: HvxVector, vu: HvxVector, vv: HvxVector) -> HvxVector {
    vdmpyhvsat_acc(vx, vu, vv)
}

/// `Vdd32.uw=vdsad(Vuu32.uh,Rt32.uh)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vdsaduh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wuw_vdsad_WuhRuh(vuu: HvxVectorPair, rt: i32) -> HvxVectorPair {
    vdsaduh(vuu, rt)
}

/// `Vxx32.uw+=vdsad(Vuu32.uh,Rt32.uh)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vdsaduh_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wuw_vdsadacc_WuwWuhRuh(
    vxx: HvxVectorPair,
    vuu: HvxVectorPair,
    rt: i32,
) -> HvxVectorPair {
    vdsaduh_acc(vxx, vuu, rt)
}

/// `Vx32.w=vinsert(Rt32)`
///
/// Instruction Type: CVI_VX_LATE
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vinsertwr))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vinsert_VwR(vx: HvxVector, rt: i32) -> HvxVector {
    vinsertwr(vx, rt)
}

/// `Vd32=vlalign(Vu32,Vv32,Rt8)`
///
/// Instruction Type: CVI_VP
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vlalignb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_V_vlalign_VVR(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVector {
    vlalignb(vu, vv, rt)
}

/// `Vd32=vlalign(Vu32,Vv32,#u3)`
///
/// Instruction Type: CVI_VP
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vlalignbi))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_V_vlalign_VVI(vu: HvxVector, vv: HvxVector, iu3: i32) -> HvxVector {
    vlalignbi(vu, vv, iu3)
}

/// `Vd32.uh=vlsr(Vu32.uh,Rt32)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vlsrh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vuh_vlsr_VuhR(vu: HvxVector, rt: i32) -> HvxVector {
    vlsrh(vu, rt)
}

/// `Vd32.h=vlsr(Vu32.h,Vv32.h)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vlsrhv))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vh_vlsr_VhVh(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vlsrhv(vu, vv)
}

/// `Vd32.uw=vlsr(Vu32.uw,Rt32)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vlsrw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vuw_vlsr_VuwR(vu: HvxVector, rt: i32) -> HvxVector {
    vlsrw(vu, rt)
}

/// `Vd32.w=vlsr(Vu32.w,Vv32.w)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vlsrwv))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vlsr_VwVw(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vlsrwv(vu, vv)
}

/// `Vd32.b=vlut32(Vu32.b,Vv32.b,Rt8)`
///
/// Instruction Type: CVI_VP
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vlutvvb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vb_vlut32_VbVbR(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVector {
    vlutvvb(vu, vv, rt)
}

/// `Vx32.b|=vlut32(Vu32.b,Vv32.b,Rt8)`
///
/// Instruction Type: CVI_VP_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vlutvvb_oracc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vb_vlut32or_VbVbVbR(
    vx: HvxVector,
    vu: HvxVector,
    vv: HvxVector,
    rt: i32,
) -> HvxVector {
    vlutvvb_oracc(vx, vu, vv, rt)
}

/// `Vdd32.h=vlut16(Vu32.b,Vv32.h,Rt8)`
///
/// Instruction Type: CVI_VP_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vlutvwh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wh_vlut16_VbVhR(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVectorPair {
    vlutvwh(vu, vv, rt)
}

/// `Vxx32.h|=vlut16(Vu32.b,Vv32.h,Rt8)`
///
/// Instruction Type: CVI_VP_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vlutvwh_oracc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wh_vlut16or_WhVbVhR(
    vxx: HvxVectorPair,
    vu: HvxVector,
    vv: HvxVector,
    rt: i32,
) -> HvxVectorPair {
    vlutvwh_oracc(vxx, vu, vv, rt)
}

/// `Vd32.h=vmax(Vu32.h,Vv32.h)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vmaxh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vh_vmax_VhVh(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmaxh(vu, vv)
}

/// `Vd32.ub=vmax(Vu32.ub,Vv32.ub)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vmaxub))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vub_vmax_VubVub(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmaxub(vu, vv)
}

/// `Vd32.uh=vmax(Vu32.uh,Vv32.uh)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vmaxuh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vuh_vmax_VuhVuh(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmaxuh(vu, vv)
}

/// `Vd32.w=vmax(Vu32.w,Vv32.w)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vmaxw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vmax_VwVw(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmaxw(vu, vv)
}

/// `Vd32.h=vmin(Vu32.h,Vv32.h)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vminh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vh_vmin_VhVh(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vminh(vu, vv)
}

/// `Vd32.ub=vmin(Vu32.ub,Vv32.ub)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vminub))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vub_vmin_VubVub(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vminub(vu, vv)
}

/// `Vd32.uh=vmin(Vu32.uh,Vv32.uh)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vminuh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vuh_vmin_VuhVuh(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vminuh(vu, vv)
}

/// `Vd32.w=vmin(Vu32.w,Vv32.w)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vminw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vmin_VwVw(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vminw(vu, vv)
}

/// `Vdd32.h=vmpa(Vuu32.ub,Rt32.b)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vmpabus))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wh_vmpa_WubRb(vuu: HvxVectorPair, rt: i32) -> HvxVectorPair {
    vmpabus(vuu, rt)
}

/// `Vxx32.h+=vmpa(Vuu32.ub,Rt32.b)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vmpabus_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wh_vmpaacc_WhWubRb(
    vxx: HvxVectorPair,
    vuu: HvxVectorPair,
    rt: i32,
) -> HvxVectorPair {
    vmpabus_acc(vxx, vuu, rt)
}

/// `Vdd32.h=vmpa(Vuu32.ub,Vvv32.b)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vmpabusv))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wh_vmpa_WubWb(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
    vmpabusv(vuu, vvv)
}

/// `Vdd32.h=vmpa(Vuu32.ub,Vvv32.ub)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vmpabuuv))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wh_vmpa_WubWub(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
    vmpabuuv(vuu, vvv)
}

/// `Vdd32.w=vmpa(Vuu32.h,Rt32.b)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vmpahb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Ww_vmpa_WhRb(vuu: HvxVectorPair, rt: i32) -> HvxVectorPair {
    vmpahb(vuu, rt)
}

/// `Vxx32.w+=vmpa(Vuu32.h,Rt32.b)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vmpahb_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Ww_vmpaacc_WwWhRb(
    vxx: HvxVectorPair,
    vuu: HvxVectorPair,
    rt: i32,
) -> HvxVectorPair {
    vmpahb_acc(vxx, vuu, rt)
}

/// `Vdd32.h=vmpy(Vu32.ub,Rt32.b)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vmpybus))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wh_vmpy_VubRb(vu: HvxVector, rt: i32) -> HvxVectorPair {
    vmpybus(vu, rt)
}

/// `Vxx32.h+=vmpy(Vu32.ub,Rt32.b)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vmpybus_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wh_vmpyacc_WhVubRb(vxx: HvxVectorPair, vu: HvxVector, rt: i32) -> HvxVectorPair {
    vmpybus_acc(vxx, vu, rt)
}

/// `Vdd32.h=vmpy(Vu32.ub,Vv32.b)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vmpybusv))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wh_vmpy_VubVb(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
    vmpybusv(vu, vv)
}

/// `Vxx32.h+=vmpy(Vu32.ub,Vv32.b)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vmpybusv_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wh_vmpyacc_WhVubVb(
    vxx: HvxVectorPair,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPair {
    vmpybusv_acc(vxx, vu, vv)
}

/// `Vdd32.h=vmpy(Vu32.b,Vv32.b)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vmpybv))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wh_vmpy_VbVb(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
    vmpybv(vu, vv)
}

/// `Vxx32.h+=vmpy(Vu32.b,Vv32.b)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vmpybv_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wh_vmpyacc_WhVbVb(
    vxx: HvxVectorPair,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPair {
    vmpybv_acc(vxx, vu, vv)
}

/// `Vd32.w=vmpye(Vu32.w,Vv32.uh)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vmpyewuh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vmpye_VwVuh(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmpyewuh(vu, vv)
}

/// `Vdd32.w=vmpy(Vu32.h,Rt32.h)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vmpyh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Ww_vmpy_VhRh(vu: HvxVector, rt: i32) -> HvxVectorPair {
    vmpyh(vu, rt)
}

/// `Vxx32.w+=vmpy(Vu32.h,Rt32.h):sat`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vmpyhsat_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Ww_vmpyacc_WwVhRh_sat(
    vxx: HvxVectorPair,
    vu: HvxVector,
    rt: i32,
) -> HvxVectorPair {
    vmpyhsat_acc(vxx, vu, rt)
}

/// `Vd32.h=vmpy(Vu32.h,Rt32.h):<<1:rnd:sat`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vmpyhsrs))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vh_vmpy_VhRh_s1_rnd_sat(vu: HvxVector, rt: i32) -> HvxVector {
    vmpyhsrs(vu, rt)
}

/// `Vd32.h=vmpy(Vu32.h,Rt32.h):<<1:sat`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vmpyhss))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vh_vmpy_VhRh_s1_sat(vu: HvxVector, rt: i32) -> HvxVector {
    vmpyhss(vu, rt)
}

/// `Vdd32.w=vmpy(Vu32.h,Vv32.uh)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vmpyhus))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Ww_vmpy_VhVuh(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
    vmpyhus(vu, vv)
}

/// `Vxx32.w+=vmpy(Vu32.h,Vv32.uh)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vmpyhus_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Ww_vmpyacc_WwVhVuh(
    vxx: HvxVectorPair,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPair {
    vmpyhus_acc(vxx, vu, vv)
}

/// `Vdd32.w=vmpy(Vu32.h,Vv32.h)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vmpyhv))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Ww_vmpy_VhVh(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
    vmpyhv(vu, vv)
}

/// `Vxx32.w+=vmpy(Vu32.h,Vv32.h)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vmpyhv_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Ww_vmpyacc_WwVhVh(
    vxx: HvxVectorPair,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPair {
    vmpyhv_acc(vxx, vu, vv)
}

/// `Vd32.h=vmpy(Vu32.h,Vv32.h):<<1:rnd:sat`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vmpyhvsrs))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vh_vmpy_VhVh_s1_rnd_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmpyhvsrs(vu, vv)
}

/// `Vd32.w=vmpyieo(Vu32.h,Vv32.h)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vmpyieoh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vmpyieo_VhVh(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmpyieoh(vu, vv)
}

/// `Vx32.w+=vmpyie(Vu32.w,Vv32.h)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vmpyiewh_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vmpyieacc_VwVwVh(vx: HvxVector, vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmpyiewh_acc(vx, vu, vv)
}

/// `Vd32.w=vmpyie(Vu32.w,Vv32.uh)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vmpyiewuh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vmpyie_VwVuh(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmpyiewuh(vu, vv)
}

/// `Vx32.w+=vmpyie(Vu32.w,Vv32.uh)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vmpyiewuh_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vmpyieacc_VwVwVuh(vx: HvxVector, vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmpyiewuh_acc(vx, vu, vv)
}

/// `Vd32.h=vmpyi(Vu32.h,Vv32.h)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vmpyih))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vh_vmpyi_VhVh(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmpyih(vu, vv)
}

/// `Vx32.h+=vmpyi(Vu32.h,Vv32.h)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vmpyih_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vh_vmpyiacc_VhVhVh(vx: HvxVector, vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmpyih_acc(vx, vu, vv)
}

/// `Vd32.h=vmpyi(Vu32.h,Rt32.b)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vmpyihb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vh_vmpyi_VhRb(vu: HvxVector, rt: i32) -> HvxVector {
    vmpyihb(vu, rt)
}

/// `Vx32.h+=vmpyi(Vu32.h,Rt32.b)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vmpyihb_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vh_vmpyiacc_VhVhRb(vx: HvxVector, vu: HvxVector, rt: i32) -> HvxVector {
    vmpyihb_acc(vx, vu, rt)
}

/// `Vd32.w=vmpyio(Vu32.w,Vv32.h)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vmpyiowh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vmpyio_VwVh(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmpyiowh(vu, vv)
}

/// `Vd32.w=vmpyi(Vu32.w,Rt32.b)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vmpyiwb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vmpyi_VwRb(vu: HvxVector, rt: i32) -> HvxVector {
    vmpyiwb(vu, rt)
}

/// `Vx32.w+=vmpyi(Vu32.w,Rt32.b)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vmpyiwb_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vmpyiacc_VwVwRb(vx: HvxVector, vu: HvxVector, rt: i32) -> HvxVector {
    vmpyiwb_acc(vx, vu, rt)
}

/// `Vd32.w=vmpyi(Vu32.w,Rt32.h)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vmpyiwh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vmpyi_VwRh(vu: HvxVector, rt: i32) -> HvxVector {
    vmpyiwh(vu, rt)
}

/// `Vx32.w+=vmpyi(Vu32.w,Rt32.h)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vmpyiwh_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vmpyiacc_VwVwRh(vx: HvxVector, vu: HvxVector, rt: i32) -> HvxVector {
    vmpyiwh_acc(vx, vu, rt)
}

/// `Vd32.w=vmpyo(Vu32.w,Vv32.h):<<1:sat`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vmpyowh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vmpyo_VwVh_s1_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmpyowh(vu, vv)
}

/// `Vd32.w=vmpyo(Vu32.w,Vv32.h):<<1:rnd:sat`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vmpyowh_rnd))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vmpyo_VwVh_s1_rnd_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmpyowh_rnd(vu, vv)
}

/// `Vx32.w+=vmpyo(Vu32.w,Vv32.h):<<1:rnd:sat:shift`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vmpyowh_rnd_sacc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vmpyoacc_VwVwVh_s1_rnd_sat_shift(
    vx: HvxVector,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVector {
    vmpyowh_rnd_sacc(vx, vu, vv)
}

/// `Vx32.w+=vmpyo(Vu32.w,Vv32.h):<<1:sat:shift`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vmpyowh_sacc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vmpyoacc_VwVwVh_s1_sat_shift(
    vx: HvxVector,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVector {
    vmpyowh_sacc(vx, vu, vv)
}

/// `Vdd32.uh=vmpy(Vu32.ub,Rt32.ub)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vmpyub))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wuh_vmpy_VubRub(vu: HvxVector, rt: i32) -> HvxVectorPair {
    vmpyub(vu, rt)
}

/// `Vxx32.uh+=vmpy(Vu32.ub,Rt32.ub)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vmpyub_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wuh_vmpyacc_WuhVubRub(
    vxx: HvxVectorPair,
    vu: HvxVector,
    rt: i32,
) -> HvxVectorPair {
    vmpyub_acc(vxx, vu, rt)
}

/// `Vdd32.uh=vmpy(Vu32.ub,Vv32.ub)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vmpyubv))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wuh_vmpy_VubVub(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
    vmpyubv(vu, vv)
}

/// `Vxx32.uh+=vmpy(Vu32.ub,Vv32.ub)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vmpyubv_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wuh_vmpyacc_WuhVubVub(
    vxx: HvxVectorPair,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPair {
    vmpyubv_acc(vxx, vu, vv)
}

/// `Vdd32.uw=vmpy(Vu32.uh,Rt32.uh)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vmpyuh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wuw_vmpy_VuhRuh(vu: HvxVector, rt: i32) -> HvxVectorPair {
    vmpyuh(vu, rt)
}

/// `Vxx32.uw+=vmpy(Vu32.uh,Rt32.uh)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vmpyuh_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wuw_vmpyacc_WuwVuhRuh(
    vxx: HvxVectorPair,
    vu: HvxVector,
    rt: i32,
) -> HvxVectorPair {
    vmpyuh_acc(vxx, vu, rt)
}

/// `Vdd32.uw=vmpy(Vu32.uh,Vv32.uh)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vmpyuhv))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wuw_vmpy_VuhVuh(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
    vmpyuhv(vu, vv)
}

/// `Vxx32.uw+=vmpy(Vu32.uh,Vv32.uh)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vmpyuhv_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wuw_vmpyacc_WuwVuhVuh(
    vxx: HvxVectorPair,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPair {
    vmpyuhv_acc(vxx, vu, vv)
}

/// `Vd32.h=vnavg(Vu32.h,Vv32.h)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vnavgh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vh_vnavg_VhVh(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vnavgh(vu, vv)
}

/// `Vd32.b=vnavg(Vu32.ub,Vv32.ub)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vnavgub))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vb_vnavg_VubVub(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vnavgub(vu, vv)
}

/// `Vd32.w=vnavg(Vu32.w,Vv32.w)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vnavgw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vnavg_VwVw(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vnavgw(vu, vv)
}

/// `Vd32.h=vnormamt(Vu32.h)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vnormamth))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vh_vnormamt_Vh(vu: HvxVector) -> HvxVector {
    vnormamth(vu)
}

/// `Vd32.w=vnormamt(Vu32.w)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vnormamtw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vnormamt_Vw(vu: HvxVector) -> HvxVector {
    vnormamtw(vu)
}

/// `Vd32=vnot(Vu32)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vnot))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_V_vnot_V(vu: HvxVector) -> HvxVector {
    vnot(vu)
}

/// `Vd32=vor(Vu32,Vv32)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vor))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_V_vor_VV(vu: HvxVector, vv: HvxVector) -> HvxVector {
    simd_or(vu, vv)
}

/// `Vd32.b=vpacke(Vu32.h,Vv32.h)`
///
/// Instruction Type: CVI_VP
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vpackeb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vb_vpacke_VhVh(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vpackeb(vu, vv)
}

/// `Vd32.h=vpacke(Vu32.w,Vv32.w)`
///
/// Instruction Type: CVI_VP
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vpackeh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vh_vpacke_VwVw(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vpackeh(vu, vv)
}

/// `Vd32.b=vpack(Vu32.h,Vv32.h):sat`
///
/// Instruction Type: CVI_VP
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vpackhb_sat))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vb_vpack_VhVh_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vpackhb_sat(vu, vv)
}

/// `Vd32.ub=vpack(Vu32.h,Vv32.h):sat`
///
/// Instruction Type: CVI_VP
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vpackhub_sat))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vub_vpack_VhVh_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vpackhub_sat(vu, vv)
}

/// `Vd32.b=vpacko(Vu32.h,Vv32.h)`
///
/// Instruction Type: CVI_VP
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vpackob))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vb_vpacko_VhVh(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vpackob(vu, vv)
}

/// `Vd32.h=vpacko(Vu32.w,Vv32.w)`
///
/// Instruction Type: CVI_VP
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vpackoh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vh_vpacko_VwVw(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vpackoh(vu, vv)
}

/// `Vd32.h=vpack(Vu32.w,Vv32.w):sat`
///
/// Instruction Type: CVI_VP
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vpackwh_sat))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vh_vpack_VwVw_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vpackwh_sat(vu, vv)
}

/// `Vd32.uh=vpack(Vu32.w,Vv32.w):sat`
///
/// Instruction Type: CVI_VP
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vpackwuh_sat))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vuh_vpack_VwVw_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vpackwuh_sat(vu, vv)
}

/// `Vd32.h=vpopcount(Vu32.h)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vpopcounth))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vh_vpopcount_Vh(vu: HvxVector) -> HvxVector {
    vpopcounth(vu)
}

/// `Vd32=vrdelta(Vu32,Vv32)`
///
/// Instruction Type: CVI_VP
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vrdelta))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_V_vrdelta_VV(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vrdelta(vu, vv)
}

/// `Vd32.w=vrmpy(Vu32.ub,Rt32.b)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vrmpybus))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vrmpy_VubRb(vu: HvxVector, rt: i32) -> HvxVector {
    vrmpybus(vu, rt)
}

/// `Vx32.w+=vrmpy(Vu32.ub,Rt32.b)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vrmpybus_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vrmpyacc_VwVubRb(vx: HvxVector, vu: HvxVector, rt: i32) -> HvxVector {
    vrmpybus_acc(vx, vu, rt)
}

/// `Vdd32.w=vrmpy(Vuu32.ub,Rt32.b,#u1)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vrmpybusi))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Ww_vrmpy_WubRbI(vuu: HvxVectorPair, rt: i32, iu1: i32) -> HvxVectorPair {
    vrmpybusi(vuu, rt, iu1)
}

/// `Vxx32.w+=vrmpy(Vuu32.ub,Rt32.b,#u1)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vrmpybusi_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Ww_vrmpyacc_WwWubRbI(
    vxx: HvxVectorPair,
    vuu: HvxVectorPair,
    rt: i32,
    iu1: i32,
) -> HvxVectorPair {
    vrmpybusi_acc(vxx, vuu, rt, iu1)
}

/// `Vd32.w=vrmpy(Vu32.ub,Vv32.b)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vrmpybusv))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vrmpy_VubVb(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vrmpybusv(vu, vv)
}

/// `Vx32.w+=vrmpy(Vu32.ub,Vv32.b)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vrmpybusv_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vrmpyacc_VwVubVb(vx: HvxVector, vu: HvxVector, vv: HvxVector) -> HvxVector {
    vrmpybusv_acc(vx, vu, vv)
}

/// `Vd32.w=vrmpy(Vu32.b,Vv32.b)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vrmpybv))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vrmpy_VbVb(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vrmpybv(vu, vv)
}

/// `Vx32.w+=vrmpy(Vu32.b,Vv32.b)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vrmpybv_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vrmpyacc_VwVbVb(vx: HvxVector, vu: HvxVector, vv: HvxVector) -> HvxVector {
    vrmpybv_acc(vx, vu, vv)
}

/// `Vd32.uw=vrmpy(Vu32.ub,Rt32.ub)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vrmpyub))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vuw_vrmpy_VubRub(vu: HvxVector, rt: i32) -> HvxVector {
    vrmpyub(vu, rt)
}

/// `Vx32.uw+=vrmpy(Vu32.ub,Rt32.ub)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vrmpyub_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vuw_vrmpyacc_VuwVubRub(vx: HvxVector, vu: HvxVector, rt: i32) -> HvxVector {
    vrmpyub_acc(vx, vu, rt)
}

/// `Vdd32.uw=vrmpy(Vuu32.ub,Rt32.ub,#u1)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vrmpyubi))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wuw_vrmpy_WubRubI(vuu: HvxVectorPair, rt: i32, iu1: i32) -> HvxVectorPair {
    vrmpyubi(vuu, rt, iu1)
}

/// `Vxx32.uw+=vrmpy(Vuu32.ub,Rt32.ub,#u1)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vrmpyubi_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wuw_vrmpyacc_WuwWubRubI(
    vxx: HvxVectorPair,
    vuu: HvxVectorPair,
    rt: i32,
    iu1: i32,
) -> HvxVectorPair {
    vrmpyubi_acc(vxx, vuu, rt, iu1)
}

/// `Vd32.uw=vrmpy(Vu32.ub,Vv32.ub)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vrmpyubv))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vuw_vrmpy_VubVub(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vrmpyubv(vu, vv)
}

/// `Vx32.uw+=vrmpy(Vu32.ub,Vv32.ub)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vrmpyubv_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vuw_vrmpyacc_VuwVubVub(vx: HvxVector, vu: HvxVector, vv: HvxVector) -> HvxVector {
    vrmpyubv_acc(vx, vu, vv)
}

/// `Vd32=vror(Vu32,Rt32)`
///
/// Instruction Type: CVI_VP
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vror))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_V_vror_VR(vu: HvxVector, rt: i32) -> HvxVector {
    vror(vu, rt)
}

/// `Vd32.b=vround(Vu32.h,Vv32.h):sat`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vroundhb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vb_vround_VhVh_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vroundhb(vu, vv)
}

/// `Vd32.ub=vround(Vu32.h,Vv32.h):sat`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vroundhub))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vub_vround_VhVh_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vroundhub(vu, vv)
}

/// `Vd32.h=vround(Vu32.w,Vv32.w):sat`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vroundwh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vh_vround_VwVw_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vroundwh(vu, vv)
}

/// `Vd32.uh=vround(Vu32.w,Vv32.w):sat`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vroundwuh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vuh_vround_VwVw_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vroundwuh(vu, vv)
}

/// `Vdd32.uw=vrsad(Vuu32.ub,Rt32.ub,#u1)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vrsadubi))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wuw_vrsad_WubRubI(vuu: HvxVectorPair, rt: i32, iu1: i32) -> HvxVectorPair {
    vrsadubi(vuu, rt, iu1)
}

/// `Vxx32.uw+=vrsad(Vuu32.ub,Rt32.ub,#u1)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vrsadubi_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wuw_vrsadacc_WuwWubRubI(
    vxx: HvxVectorPair,
    vuu: HvxVectorPair,
    rt: i32,
    iu1: i32,
) -> HvxVectorPair {
    vrsadubi_acc(vxx, vuu, rt, iu1)
}

/// `Vd32.ub=vsat(Vu32.h,Vv32.h)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vsathub))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vub_vsat_VhVh(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vsathub(vu, vv)
}

/// `Vd32.h=vsat(Vu32.w,Vv32.w)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vsatwh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vh_vsat_VwVw(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vsatwh(vu, vv)
}

/// `Vdd32.h=vsxt(Vu32.b)`
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vsb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wh_vsxt_Vb(vu: HvxVector) -> HvxVectorPair {
    vsb(vu)
}

/// `Vdd32.w=vsxt(Vu32.h)`
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vsh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Ww_vsxt_Vh(vu: HvxVector) -> HvxVectorPair {
    vsh(vu)
}

/// `Vd32.h=vshuffe(Vu32.h,Vv32.h)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vshufeh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vh_vshuffe_VhVh(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vshufeh(vu, vv)
}

/// `Vd32.b=vshuff(Vu32.b)`
///
/// Instruction Type: CVI_VP
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vshuffb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vb_vshuff_Vb(vu: HvxVector) -> HvxVector {
    vshuffb(vu)
}

/// `Vd32.b=vshuffe(Vu32.b,Vv32.b)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vshuffeb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vb_vshuffe_VbVb(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vshuffeb(vu, vv)
}

/// `Vd32.h=vshuff(Vu32.h)`
///
/// Instruction Type: CVI_VP
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vshuffh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vh_vshuff_Vh(vu: HvxVector) -> HvxVector {
    vshuffh(vu)
}

/// `Vd32.b=vshuffo(Vu32.b,Vv32.b)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vshuffob))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vb_vshuffo_VbVb(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vshuffob(vu, vv)
}

/// `Vdd32=vshuff(Vu32,Vv32,Rt8)`
///
/// Instruction Type: CVI_VP_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vshuffvdd))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_W_vshuff_VVR(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVectorPair {
    vshuffvdd(vu, vv, rt)
}

/// `Vdd32.b=vshuffoe(Vu32.b,Vv32.b)`
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vshufoeb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wb_vshuffoe_VbVb(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
    vshufoeb(vu, vv)
}

/// `Vdd32.h=vshuffoe(Vu32.h,Vv32.h)`
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vshufoeh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wh_vshuffoe_VhVh(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
    vshufoeh(vu, vv)
}

/// `Vd32.h=vshuffo(Vu32.h,Vv32.h)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vshufoh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vh_vshuffo_VhVh(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vshufoh(vu, vv)
}

/// `Vd32.b=vsub(Vu32.b,Vv32.b)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vsubb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vb_vsub_VbVb(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vsubb(vu, vv)
}

/// `Vdd32.b=vsub(Vuu32.b,Vvv32.b)`
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vsubb_dv))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wb_vsub_WbWb(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
    vsubb_dv(vuu, vvv)
}

/// `Vd32.h=vsub(Vu32.h,Vv32.h)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vsubh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vh_vsub_VhVh(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vsubh(vu, vv)
}

/// `Vdd32.h=vsub(Vuu32.h,Vvv32.h)`
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vsubh_dv))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wh_vsub_WhWh(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
    vsubh_dv(vuu, vvv)
}

/// `Vd32.h=vsub(Vu32.h,Vv32.h):sat`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vsubhsat))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vh_vsub_VhVh_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vsubhsat(vu, vv)
}

/// `Vdd32.h=vsub(Vuu32.h,Vvv32.h):sat`
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vsubhsat_dv))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wh_vsub_WhWh_sat(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
    vsubhsat_dv(vuu, vvv)
}

/// `Vdd32.w=vsub(Vu32.h,Vv32.h)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vsubhw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Ww_vsub_VhVh(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
    vsubhw(vu, vv)
}

/// `Vdd32.h=vsub(Vu32.ub,Vv32.ub)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vsububh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wh_vsub_VubVub(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
    vsububh(vu, vv)
}

/// `Vd32.ub=vsub(Vu32.ub,Vv32.ub):sat`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vsububsat))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vub_vsub_VubVub_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vsububsat(vu, vv)
}

/// `Vdd32.ub=vsub(Vuu32.ub,Vvv32.ub):sat`
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vsububsat_dv))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wub_vsub_WubWub_sat(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
    vsububsat_dv(vuu, vvv)
}

/// `Vd32.uh=vsub(Vu32.uh,Vv32.uh):sat`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vsubuhsat))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vuh_vsub_VuhVuh_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vsubuhsat(vu, vv)
}

/// `Vdd32.uh=vsub(Vuu32.uh,Vvv32.uh):sat`
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vsubuhsat_dv))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wuh_vsub_WuhWuh_sat(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
    vsubuhsat_dv(vuu, vvv)
}

/// `Vdd32.w=vsub(Vu32.uh,Vv32.uh)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vsubuhw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Ww_vsub_VuhVuh(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
    vsubuhw(vu, vv)
}

/// `Vd32.w=vsub(Vu32.w,Vv32.w)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vsubw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vsub_VwVw(vu: HvxVector, vv: HvxVector) -> HvxVector {
    simd_sub(vu, vv)
}

/// `Vdd32.w=vsub(Vuu32.w,Vvv32.w)`
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vsubw_dv))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Ww_vsub_WwWw(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
    vsubw_dv(vuu, vvv)
}

/// `Vd32.w=vsub(Vu32.w,Vv32.w):sat`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vsubwsat))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vsub_VwVw_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vsubwsat(vu, vv)
}

/// `Vdd32.w=vsub(Vuu32.w,Vvv32.w):sat`
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vsubwsat_dv))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Ww_vsub_WwWw_sat(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
    vsubwsat_dv(vuu, vvv)
}

/// `Vdd32.h=vtmpy(Vuu32.b,Rt32.b)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vtmpyb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wh_vtmpy_WbRb(vuu: HvxVectorPair, rt: i32) -> HvxVectorPair {
    vtmpyb(vuu, rt)
}

/// `Vxx32.h+=vtmpy(Vuu32.b,Rt32.b)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vtmpyb_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wh_vtmpyacc_WhWbRb(
    vxx: HvxVectorPair,
    vuu: HvxVectorPair,
    rt: i32,
) -> HvxVectorPair {
    vtmpyb_acc(vxx, vuu, rt)
}

/// `Vdd32.h=vtmpy(Vuu32.ub,Rt32.b)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vtmpybus))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wh_vtmpy_WubRb(vuu: HvxVectorPair, rt: i32) -> HvxVectorPair {
    vtmpybus(vuu, rt)
}

/// `Vxx32.h+=vtmpy(Vuu32.ub,Rt32.b)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vtmpybus_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wh_vtmpyacc_WhWubRb(
    vxx: HvxVectorPair,
    vuu: HvxVectorPair,
    rt: i32,
) -> HvxVectorPair {
    vtmpybus_acc(vxx, vuu, rt)
}

/// `Vdd32.w=vtmpy(Vuu32.h,Rt32.b)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vtmpyhb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Ww_vtmpy_WhRb(vuu: HvxVectorPair, rt: i32) -> HvxVectorPair {
    vtmpyhb(vuu, rt)
}

/// `Vxx32.w+=vtmpy(Vuu32.h,Rt32.b)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vtmpyhb_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Ww_vtmpyacc_WwWhRb(
    vxx: HvxVectorPair,
    vuu: HvxVectorPair,
    rt: i32,
) -> HvxVectorPair {
    vtmpyhb_acc(vxx, vuu, rt)
}

/// `Vdd32.h=vunpack(Vu32.b)`
///
/// Instruction Type: CVI_VP_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vunpackb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wh_vunpack_Vb(vu: HvxVector) -> HvxVectorPair {
    vunpackb(vu)
}

/// `Vdd32.w=vunpack(Vu32.h)`
///
/// Instruction Type: CVI_VP_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vunpackh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Ww_vunpack_Vh(vu: HvxVector) -> HvxVectorPair {
    vunpackh(vu)
}

/// `Vxx32.h|=vunpacko(Vu32.b)`
///
/// Instruction Type: CVI_VP_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vunpackob))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wh_vunpackoor_WhVb(vxx: HvxVectorPair, vu: HvxVector) -> HvxVectorPair {
    vunpackob(vxx, vu)
}

/// `Vxx32.w|=vunpacko(Vu32.h)`
///
/// Instruction Type: CVI_VP_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vunpackoh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Ww_vunpackoor_WwVh(vxx: HvxVectorPair, vu: HvxVector) -> HvxVectorPair {
    vunpackoh(vxx, vu)
}

/// `Vdd32.uh=vunpack(Vu32.ub)`
///
/// Instruction Type: CVI_VP_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vunpackub))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wuh_vunpack_Vub(vu: HvxVector) -> HvxVectorPair {
    vunpackub(vu)
}

/// `Vdd32.uw=vunpack(Vu32.uh)`
///
/// Instruction Type: CVI_VP_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vunpackuh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wuw_vunpack_Vuh(vu: HvxVector) -> HvxVectorPair {
    vunpackuh(vu)
}

/// `Vd32=vxor(Vu32,Vv32)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vxor))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_V_vxor_VV(vu: HvxVector, vv: HvxVector) -> HvxVector {
    simd_xor(vu, vv)
}

/// `Vdd32.uh=vzxt(Vu32.ub)`
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vzb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wuh_vzxt_Vub(vu: HvxVector) -> HvxVectorPair {
    vzb(vu)
}

/// `Vdd32.uw=vzxt(Vu32.uh)`
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[cfg_attr(test, assert_instr(vzh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wuw_vzxt_Vuh(vu: HvxVector) -> HvxVectorPair {
    vzh(vu)
}

/// `Vd32.b=vsplat(Rt32)`
///
/// Instruction Type: CVI_VX_LATE
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[cfg_attr(test, assert_instr(lvsplatb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vb_vsplat_R(rt: i32) -> HvxVector {
    lvsplatb(rt)
}

/// `Vd32.h=vsplat(Rt32)`
///
/// Instruction Type: CVI_VX_LATE
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[cfg_attr(test, assert_instr(lvsplath))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vh_vsplat_R(rt: i32) -> HvxVector {
    lvsplath(rt)
}

/// `Vd32.b=vadd(Vu32.b,Vv32.b):sat`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[cfg_attr(test, assert_instr(vaddbsat))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vb_vadd_VbVb_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vaddbsat(vu, vv)
}

/// `Vdd32.b=vadd(Vuu32.b,Vvv32.b):sat`
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[cfg_attr(test, assert_instr(vaddbsat_dv))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wb_vadd_WbWb_sat(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
    vaddbsat_dv(vuu, vvv)
}

/// `Vd32.h=vadd(vclb(Vu32.h),Vv32.h)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[cfg_attr(test, assert_instr(vaddclbh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vh_vadd_vclb_VhVh(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vaddclbh(vu, vv)
}

/// `Vd32.w=vadd(vclb(Vu32.w),Vv32.w)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[cfg_attr(test, assert_instr(vaddclbw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vadd_vclb_VwVw(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vaddclbw(vu, vv)
}

/// `Vxx32.w+=vadd(Vu32.h,Vv32.h)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[cfg_attr(test, assert_instr(vaddhw_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Ww_vaddacc_WwVhVh(
    vxx: HvxVectorPair,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPair {
    vaddhw_acc(vxx, vu, vv)
}

/// `Vxx32.h+=vadd(Vu32.ub,Vv32.ub)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[cfg_attr(test, assert_instr(vaddubh_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wh_vaddacc_WhVubVub(
    vxx: HvxVectorPair,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPair {
    vaddubh_acc(vxx, vu, vv)
}

/// `Vd32.ub=vadd(Vu32.ub,Vv32.b):sat`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[cfg_attr(test, assert_instr(vaddububb_sat))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vub_vadd_VubVb_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vaddububb_sat(vu, vv)
}

/// `Vxx32.w+=vadd(Vu32.uh,Vv32.uh)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[cfg_attr(test, assert_instr(vadduhw_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Ww_vaddacc_WwVuhVuh(
    vxx: HvxVectorPair,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPair {
    vadduhw_acc(vxx, vu, vv)
}

/// `Vd32.uw=vadd(Vu32.uw,Vv32.uw):sat`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[cfg_attr(test, assert_instr(vadduwsat))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vuw_vadd_VuwVuw_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vadduwsat(vu, vv)
}

/// `Vdd32.uw=vadd(Vuu32.uw,Vvv32.uw):sat`
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[cfg_attr(test, assert_instr(vadduwsat_dv))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wuw_vadd_WuwWuw_sat(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
    vadduwsat_dv(vuu, vvv)
}

/// `Vd32.b=vasr(Vu32.h,Vv32.h,Rt8):sat`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[cfg_attr(test, assert_instr(vasrhbsat))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vb_vasr_VhVhR_sat(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVector {
    vasrhbsat(vu, vv, rt)
}

/// `Vd32.uh=vasr(Vu32.uw,Vv32.uw,Rt8):rnd:sat`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[cfg_attr(test, assert_instr(vasruwuhrndsat))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vuh_vasr_VuwVuwR_rnd_sat(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVector {
    vasruwuhrndsat(vu, vv, rt)
}

/// `Vd32.uh=vasr(Vu32.w,Vv32.w,Rt8):rnd:sat`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[cfg_attr(test, assert_instr(vasrwuhrndsat))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vuh_vasr_VwVwR_rnd_sat(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVector {
    vasrwuhrndsat(vu, vv, rt)
}

/// `Vd32.ub=vlsr(Vu32.ub,Rt32)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[cfg_attr(test, assert_instr(vlsrb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vub_vlsr_VubR(vu: HvxVector, rt: i32) -> HvxVector {
    vlsrb(vu, rt)
}

/// `Vd32.b=vlut32(Vu32.b,Vv32.b,Rt8):nomatch`
///
/// Instruction Type: CVI_VP
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[cfg_attr(test, assert_instr(vlutvvb_nm))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vb_vlut32_VbVbR_nomatch(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVector {
    vlutvvb_nm(vu, vv, rt)
}

/// `Vx32.b|=vlut32(Vu32.b,Vv32.b,#u3)`
///
/// Instruction Type: CVI_VP_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[cfg_attr(test, assert_instr(vlutvvb_oracci))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vb_vlut32or_VbVbVbI(
    vx: HvxVector,
    vu: HvxVector,
    vv: HvxVector,
    iu3: i32,
) -> HvxVector {
    vlutvvb_oracci(vx, vu, vv, iu3)
}

/// `Vd32.b=vlut32(Vu32.b,Vv32.b,#u3)`
///
/// Instruction Type: CVI_VP
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[cfg_attr(test, assert_instr(vlutvvbi))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vb_vlut32_VbVbI(vu: HvxVector, vv: HvxVector, iu3: i32) -> HvxVector {
    vlutvvbi(vu, vv, iu3)
}

/// `Vdd32.h=vlut16(Vu32.b,Vv32.h,Rt8):nomatch`
///
/// Instruction Type: CVI_VP_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[cfg_attr(test, assert_instr(vlutvwh_nm))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wh_vlut16_VbVhR_nomatch(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVectorPair {
    vlutvwh_nm(vu, vv, rt)
}

/// `Vxx32.h|=vlut16(Vu32.b,Vv32.h,#u3)`
///
/// Instruction Type: CVI_VP_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[cfg_attr(test, assert_instr(vlutvwh_oracci))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wh_vlut16or_WhVbVhI(
    vxx: HvxVectorPair,
    vu: HvxVector,
    vv: HvxVector,
    iu3: i32,
) -> HvxVectorPair {
    vlutvwh_oracci(vxx, vu, vv, iu3)
}

/// `Vdd32.h=vlut16(Vu32.b,Vv32.h,#u3)`
///
/// Instruction Type: CVI_VP_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[cfg_attr(test, assert_instr(vlutvwhi))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wh_vlut16_VbVhI(vu: HvxVector, vv: HvxVector, iu3: i32) -> HvxVectorPair {
    vlutvwhi(vu, vv, iu3)
}

/// `Vd32.b=vmax(Vu32.b,Vv32.b)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[cfg_attr(test, assert_instr(vmaxb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vb_vmax_VbVb(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmaxb(vu, vv)
}

/// `Vd32.b=vmin(Vu32.b,Vv32.b)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[cfg_attr(test, assert_instr(vminb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vb_vmin_VbVb(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vminb(vu, vv)
}

/// `Vdd32.w=vmpa(Vuu32.uh,Rt32.b)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[cfg_attr(test, assert_instr(vmpauhb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Ww_vmpa_WuhRb(vuu: HvxVectorPair, rt: i32) -> HvxVectorPair {
    vmpauhb(vuu, rt)
}

/// `Vxx32.w+=vmpa(Vuu32.uh,Rt32.b)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[cfg_attr(test, assert_instr(vmpauhb_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Ww_vmpaacc_WwWuhRb(
    vxx: HvxVectorPair,
    vuu: HvxVectorPair,
    rt: i32,
) -> HvxVectorPair {
    vmpauhb_acc(vxx, vuu, rt)
}

/// `Vdd32=vmpye(Vu32.w,Vv32.uh)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[cfg_attr(test, assert_instr(vmpyewuh_64))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_W_vmpye_VwVuh(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
    vmpyewuh_64(vu, vv)
}

/// `Vd32.w=vmpyi(Vu32.w,Rt32.ub)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[cfg_attr(test, assert_instr(vmpyiwub))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vmpyi_VwRub(vu: HvxVector, rt: i32) -> HvxVector {
    vmpyiwub(vu, rt)
}

/// `Vx32.w+=vmpyi(Vu32.w,Rt32.ub)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[cfg_attr(test, assert_instr(vmpyiwub_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vmpyiacc_VwVwRub(vx: HvxVector, vu: HvxVector, rt: i32) -> HvxVector {
    vmpyiwub_acc(vx, vu, rt)
}

/// `Vxx32+=vmpyo(Vu32.w,Vv32.h)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[cfg_attr(test, assert_instr(vmpyowh_64_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_W_vmpyoacc_WVwVh(
    vxx: HvxVectorPair,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPair {
    vmpyowh_64_acc(vxx, vu, vv)
}

/// `Vd32.ub=vround(Vu32.uh,Vv32.uh):sat`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[cfg_attr(test, assert_instr(vrounduhub))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vub_vround_VuhVuh_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vrounduhub(vu, vv)
}

/// `Vd32.uh=vround(Vu32.uw,Vv32.uw):sat`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[cfg_attr(test, assert_instr(vrounduwuh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vuh_vround_VuwVuw_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vrounduwuh(vu, vv)
}

/// `Vd32.uh=vsat(Vu32.uw,Vv32.uw)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[cfg_attr(test, assert_instr(vsatuwuh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vuh_vsat_VuwVuw(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vsatuwuh(vu, vv)
}

/// `Vd32.b=vsub(Vu32.b,Vv32.b):sat`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[cfg_attr(test, assert_instr(vsubbsat))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vb_vsub_VbVb_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vsubbsat(vu, vv)
}

/// `Vdd32.b=vsub(Vuu32.b,Vvv32.b):sat`
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[cfg_attr(test, assert_instr(vsubbsat_dv))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wb_vsub_WbWb_sat(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
    vsubbsat_dv(vuu, vvv)
}

/// `Vd32.ub=vsub(Vu32.ub,Vv32.b):sat`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[cfg_attr(test, assert_instr(vsubububb_sat))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vub_vsub_VubVb_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vsubububb_sat(vu, vv)
}

/// `Vd32.uw=vsub(Vu32.uw,Vv32.uw):sat`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[cfg_attr(test, assert_instr(vsubuwsat))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vuw_vsub_VuwVuw_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vsubuwsat(vu, vv)
}

/// `Vdd32.uw=vsub(Vuu32.uw,Vvv32.uw):sat`
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[cfg_attr(test, assert_instr(vsubuwsat_dv))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wuw_vsub_WuwWuw_sat(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
    vsubuwsat_dv(vuu, vvv)
}

/// `Vd32.b=vabs(Vu32.b)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[cfg_attr(test, assert_instr(vabsb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vb_vabs_Vb(vu: HvxVector) -> HvxVector {
    vabsb(vu)
}

/// `Vd32.b=vabs(Vu32.b):sat`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[cfg_attr(test, assert_instr(vabsb_sat))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vb_vabs_Vb_sat(vu: HvxVector) -> HvxVector {
    vabsb_sat(vu)
}

/// `Vx32.h+=vasl(Vu32.h,Rt32)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[cfg_attr(test, assert_instr(vaslh_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vh_vaslacc_VhVhR(vx: HvxVector, vu: HvxVector, rt: i32) -> HvxVector {
    vaslh_acc(vx, vu, rt)
}

/// `Vx32.h+=vasr(Vu32.h,Rt32)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[cfg_attr(test, assert_instr(vasrh_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vh_vasracc_VhVhR(vx: HvxVector, vu: HvxVector, rt: i32) -> HvxVector {
    vasrh_acc(vx, vu, rt)
}

/// `Vd32.ub=vasr(Vu32.uh,Vv32.uh,Rt8):rnd:sat`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[cfg_attr(test, assert_instr(vasruhubrndsat))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vub_vasr_VuhVuhR_rnd_sat(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVector {
    vasruhubrndsat(vu, vv, rt)
}

/// `Vd32.ub=vasr(Vu32.uh,Vv32.uh,Rt8):sat`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[cfg_attr(test, assert_instr(vasruhubsat))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vub_vasr_VuhVuhR_sat(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVector {
    vasruhubsat(vu, vv, rt)
}

/// `Vd32.uh=vasr(Vu32.uw,Vv32.uw,Rt8):sat`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[cfg_attr(test, assert_instr(vasruwuhsat))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vuh_vasr_VuwVuwR_sat(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVector {
    vasruwuhsat(vu, vv, rt)
}

/// `Vd32.b=vavg(Vu32.b,Vv32.b)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[cfg_attr(test, assert_instr(vavgb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vb_vavg_VbVb(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vavgb(vu, vv)
}

/// `Vd32.b=vavg(Vu32.b,Vv32.b):rnd`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[cfg_attr(test, assert_instr(vavgbrnd))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vb_vavg_VbVb_rnd(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vavgbrnd(vu, vv)
}

/// `Vd32.uw=vavg(Vu32.uw,Vv32.uw)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[cfg_attr(test, assert_instr(vavguw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vuw_vavg_VuwVuw(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vavguw(vu, vv)
}

/// `Vd32.uw=vavg(Vu32.uw,Vv32.uw):rnd`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[cfg_attr(test, assert_instr(vavguwrnd))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vuw_vavg_VuwVuw_rnd(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vavguwrnd(vu, vv)
}

/// `Vdd32=#0`
///
/// Instruction Type: MAPPING
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[cfg_attr(test, assert_instr(vdd0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_W_vzero() -> HvxVectorPair {
    vdd0()
}

/// `vtmp.h=vgather(Rt32,Mu2,Vv32.h).h`
///
/// Instruction Type: CVI_GATHER
/// Execution Slots: SLOT01
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[cfg_attr(test, assert_instr(vgathermh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_vgather_ARMVh(rs: *mut HvxVector, rt: i32, mu: i32, vv: HvxVector) {
    vgathermh(rs, rt, mu, vv)
}

/// `vtmp.h=vgather(Rt32,Mu2,Vvv32.w).h`
///
/// Instruction Type: CVI_GATHER_DV
/// Execution Slots: SLOT01
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[cfg_attr(test, assert_instr(vgathermhw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_vgather_ARMWw(rs: *mut HvxVector, rt: i32, mu: i32, vvv: HvxVectorPair) {
    vgathermhw(rs, rt, mu, vvv)
}

/// `vtmp.w=vgather(Rt32,Mu2,Vv32.w).w`
///
/// Instruction Type: CVI_GATHER
/// Execution Slots: SLOT01
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[cfg_attr(test, assert_instr(vgathermw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_vgather_ARMVw(rs: *mut HvxVector, rt: i32, mu: i32, vv: HvxVector) {
    vgathermw(rs, rt, mu, vv)
}

/// `Vdd32.h=vmpa(Vuu32.ub,Rt32.ub)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[cfg_attr(test, assert_instr(vmpabuu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wh_vmpa_WubRub(vuu: HvxVectorPair, rt: i32) -> HvxVectorPair {
    vmpabuu(vuu, rt)
}

/// `Vxx32.h+=vmpa(Vuu32.ub,Rt32.ub)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[cfg_attr(test, assert_instr(vmpabuu_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wh_vmpaacc_WhWubRub(
    vxx: HvxVectorPair,
    vuu: HvxVectorPair,
    rt: i32,
) -> HvxVectorPair {
    vmpabuu_acc(vxx, vuu, rt)
}

/// `Vxx32.w+=vmpy(Vu32.h,Rt32.h)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[cfg_attr(test, assert_instr(vmpyh_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Ww_vmpyacc_WwVhRh(vxx: HvxVectorPair, vu: HvxVector, rt: i32) -> HvxVectorPair {
    vmpyh_acc(vxx, vu, rt)
}

/// `Vd32.uw=vmpye(Vu32.uh,Rt32.uh)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[cfg_attr(test, assert_instr(vmpyuhe))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vuw_vmpye_VuhRuh(vu: HvxVector, rt: i32) -> HvxVector {
    vmpyuhe(vu, rt)
}

/// `Vx32.uw+=vmpye(Vu32.uh,Rt32.uh)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[cfg_attr(test, assert_instr(vmpyuhe_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vuw_vmpyeacc_VuwVuhRuh(vx: HvxVector, vu: HvxVector, rt: i32) -> HvxVector {
    vmpyuhe_acc(vx, vu, rt)
}

/// `Vd32.b=vnavg(Vu32.b,Vv32.b)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[cfg_attr(test, assert_instr(vnavgb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vb_vnavg_VbVb(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vnavgb(vu, vv)
}

/// `vscatter(Rt32,Mu2,Vv32.h).h=Vw32`
///
/// Instruction Type: CVI_SCATTER
/// Execution Slots: SLOT0
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[cfg_attr(test, assert_instr(vscattermh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_vscatter_RMVhV(rt: i32, mu: i32, vv: HvxVector, vw: HvxVector) {
    vscattermh(rt, mu, vv, vw)
}

/// `vscatter(Rt32,Mu2,Vv32.h).h+=Vw32`
///
/// Instruction Type: CVI_SCATTER
/// Execution Slots: SLOT0
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[cfg_attr(test, assert_instr(vscattermh_add))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_vscatteracc_RMVhV(rt: i32, mu: i32, vv: HvxVector, vw: HvxVector) {
    vscattermh_add(rt, mu, vv, vw)
}

/// `vscatter(Rt32,Mu2,Vvv32.w).h=Vw32`
///
/// Instruction Type: CVI_SCATTER_DV
/// Execution Slots: SLOT0
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[cfg_attr(test, assert_instr(vscattermhw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_vscatter_RMWwV(rt: i32, mu: i32, vvv: HvxVectorPair, vw: HvxVector) {
    vscattermhw(rt, mu, vvv, vw)
}

/// `vscatter(Rt32,Mu2,Vvv32.w).h+=Vw32`
///
/// Instruction Type: CVI_SCATTER_DV
/// Execution Slots: SLOT0
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[cfg_attr(test, assert_instr(vscattermhw_add))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_vscatteracc_RMWwV(rt: i32, mu: i32, vvv: HvxVectorPair, vw: HvxVector) {
    vscattermhw_add(rt, mu, vvv, vw)
}

/// `vscatter(Rt32,Mu2,Vv32.w).w=Vw32`
///
/// Instruction Type: CVI_SCATTER
/// Execution Slots: SLOT0
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[cfg_attr(test, assert_instr(vscattermw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_vscatter_RMVwV(rt: i32, mu: i32, vv: HvxVector, vw: HvxVector) {
    vscattermw(rt, mu, vv, vw)
}

/// `vscatter(Rt32,Mu2,Vv32.w).w+=Vw32`
///
/// Instruction Type: CVI_SCATTER
/// Execution Slots: SLOT0
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[cfg_attr(test, assert_instr(vscattermw_add))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_vscatteracc_RMVwV(rt: i32, mu: i32, vv: HvxVector, vw: HvxVector) {
    vscattermw_add(rt, mu, vv, vw)
}

/// `Vxx32.w=vasrinto(Vu32.w,Vv32.w)`
///
/// Instruction Type: CVI_VP_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv66"))]
#[cfg_attr(test, assert_instr(vasr_into))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Ww_vasrinto_WwVwVw(
    vxx: HvxVectorPair,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPair {
    vasr_into(vxx, vu, vv)
}

/// `Vd32.uw=vrotr(Vu32.uw,Vv32.uw)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv66"))]
#[cfg_attr(test, assert_instr(vrotr))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vuw_vrotr_VuwVuw(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vrotr(vu, vv)
}

/// `Vd32.w=vsatdw(Vu32.w,Vv32.w)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv66"))]
#[cfg_attr(test, assert_instr(vsatdw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vsatdw_VwVw(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vsatdw(vu, vv)
}

/// `Vdd32.w=v6mpy(Vuu32.ub,Vvv32.b,#u2):h`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(v6mpyhubs10))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Ww_v6mpy_WubWbI_h(
    vuu: HvxVectorPair,
    vvv: HvxVectorPair,
    iu2: i32,
) -> HvxVectorPair {
    v6mpyhubs10(vuu, vvv, iu2)
}

/// `Vxx32.w+=v6mpy(Vuu32.ub,Vvv32.b,#u2):h`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(v6mpyhubs10_vxx))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Ww_v6mpyacc_WwWubWbI_h(
    vxx: HvxVectorPair,
    vuu: HvxVectorPair,
    vvv: HvxVectorPair,
    iu2: i32,
) -> HvxVectorPair {
    v6mpyhubs10_vxx(vxx, vuu, vvv, iu2)
}

/// `Vdd32.w=v6mpy(Vuu32.ub,Vvv32.b,#u2):v`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(v6mpyvubs10))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Ww_v6mpy_WubWbI_v(
    vuu: HvxVectorPair,
    vvv: HvxVectorPair,
    iu2: i32,
) -> HvxVectorPair {
    v6mpyvubs10(vuu, vvv, iu2)
}

/// `Vxx32.w+=v6mpy(Vuu32.ub,Vvv32.b,#u2):v`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(v6mpyvubs10_vxx))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Ww_v6mpyacc_WwWubWbI_v(
    vxx: HvxVectorPair,
    vuu: HvxVectorPair,
    vvv: HvxVectorPair,
    iu2: i32,
) -> HvxVectorPair {
    v6mpyvubs10_vxx(vxx, vuu, vvv, iu2)
}

/// `Vd32.hf=vabs(Vu32.hf)`
///
/// Instruction Type: CVI_VX_LATE
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vabs_hf))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vhf_vabs_Vhf(vu: HvxVector) -> HvxVector {
    vabs_hf(vu)
}

/// `Vd32.sf=vabs(Vu32.sf)`
///
/// Instruction Type: CVI_VX_LATE
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vabs_sf))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vsf_vabs_Vsf(vu: HvxVector) -> HvxVector {
    vabs_sf(vu)
}

/// `Vd32.qf16=vadd(Vu32.hf,Vv32.hf)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vadd_hf))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vqf16_vadd_VhfVhf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vadd_hf(vu, vv)
}

/// `Vd32.hf=vadd(Vu32.hf,Vv32.hf)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vadd_hf_hf))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vhf_vadd_VhfVhf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vadd_hf_hf(vu, vv)
}

/// `Vd32.qf16=vadd(Vu32.qf16,Vv32.qf16)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vadd_qf16))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vqf16_vadd_Vqf16Vqf16(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vadd_qf16(vu, vv)
}

/// `Vd32.qf16=vadd(Vu32.qf16,Vv32.hf)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vadd_qf16_mix))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vqf16_vadd_Vqf16Vhf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vadd_qf16_mix(vu, vv)
}

/// `Vd32.qf32=vadd(Vu32.qf32,Vv32.qf32)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vadd_qf32))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vqf32_vadd_Vqf32Vqf32(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vadd_qf32(vu, vv)
}

/// `Vd32.qf32=vadd(Vu32.qf32,Vv32.sf)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vadd_qf32_mix))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vqf32_vadd_Vqf32Vsf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vadd_qf32_mix(vu, vv)
}

/// `Vd32.qf32=vadd(Vu32.sf,Vv32.sf)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vadd_sf))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vqf32_vadd_VsfVsf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vadd_sf(vu, vv)
}

/// `Vdd32.sf=vadd(Vu32.hf,Vv32.hf)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vadd_sf_hf))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wsf_vadd_VhfVhf(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
    vadd_sf_hf(vu, vv)
}

/// `Vd32.sf=vadd(Vu32.sf,Vv32.sf)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vadd_sf_sf))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vsf_vadd_VsfVsf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vadd_sf_sf(vu, vv)
}

/// `Vd32.w=vfmv(Vu32.w)`
///
/// Instruction Type: CVI_VX_LATE
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vassign_fp))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vfmv_Vw(vu: HvxVector) -> HvxVector {
    vassign_fp(vu)
}

/// `Vd32.hf=Vu32.qf16`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vconv_hf_qf16))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vhf_equals_Vqf16(vu: HvxVector) -> HvxVector {
    vconv_hf_qf16(vu)
}

/// `Vd32.hf=Vuu32.qf32`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vconv_hf_qf32))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vhf_equals_Wqf32(vuu: HvxVectorPair) -> HvxVector {
    vconv_hf_qf32(vuu)
}

/// `Vd32.sf=Vu32.qf32`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vconv_sf_qf32))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vsf_equals_Vqf32(vu: HvxVector) -> HvxVector {
    vconv_sf_qf32(vu)
}

/// `Vd32.b=vcvt(Vu32.hf,Vv32.hf)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vcvt_b_hf))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vb_vcvt_VhfVhf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vcvt_b_hf(vu, vv)
}

/// `Vd32.h=vcvt(Vu32.hf)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vcvt_h_hf))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vh_vcvt_Vhf(vu: HvxVector) -> HvxVector {
    vcvt_h_hf(vu)
}

/// `Vdd32.hf=vcvt(Vu32.b)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vcvt_hf_b))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Whf_vcvt_Vb(vu: HvxVector) -> HvxVectorPair {
    vcvt_hf_b(vu)
}

/// `Vd32.hf=vcvt(Vu32.h)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vcvt_hf_h))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vhf_vcvt_Vh(vu: HvxVector) -> HvxVector {
    vcvt_hf_h(vu)
}

/// `Vd32.hf=vcvt(Vu32.sf,Vv32.sf)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vcvt_hf_sf))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vhf_vcvt_VsfVsf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vcvt_hf_sf(vu, vv)
}

/// `Vdd32.hf=vcvt(Vu32.ub)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vcvt_hf_ub))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Whf_vcvt_Vub(vu: HvxVector) -> HvxVectorPair {
    vcvt_hf_ub(vu)
}

/// `Vd32.hf=vcvt(Vu32.uh)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vcvt_hf_uh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vhf_vcvt_Vuh(vu: HvxVector) -> HvxVector {
    vcvt_hf_uh(vu)
}

/// `Vdd32.sf=vcvt(Vu32.hf)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vcvt_sf_hf))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wsf_vcvt_Vhf(vu: HvxVector) -> HvxVectorPair {
    vcvt_sf_hf(vu)
}

/// `Vd32.ub=vcvt(Vu32.hf,Vv32.hf)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vcvt_ub_hf))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vub_vcvt_VhfVhf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vcvt_ub_hf(vu, vv)
}

/// `Vd32.uh=vcvt(Vu32.hf)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vcvt_uh_hf))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vuh_vcvt_Vhf(vu: HvxVector) -> HvxVector {
    vcvt_uh_hf(vu)
}

/// `Vd32.sf=vdmpy(Vu32.hf,Vv32.hf)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vdmpy_sf_hf))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vsf_vdmpy_VhfVhf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vdmpy_sf_hf(vu, vv)
}

/// `Vx32.sf+=vdmpy(Vu32.hf,Vv32.hf)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vdmpy_sf_hf_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vsf_vdmpyacc_VsfVhfVhf(vx: HvxVector, vu: HvxVector, vv: HvxVector) -> HvxVector {
    vdmpy_sf_hf_acc(vx, vu, vv)
}

/// `Vd32.hf=vfmax(Vu32.hf,Vv32.hf)`
///
/// Instruction Type: CVI_VX_LATE
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vfmax_hf))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vhf_vfmax_VhfVhf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vfmax_hf(vu, vv)
}

/// `Vd32.sf=vfmax(Vu32.sf,Vv32.sf)`
///
/// Instruction Type: CVI_VX_LATE
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vfmax_sf))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vsf_vfmax_VsfVsf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vfmax_sf(vu, vv)
}

/// `Vd32.hf=vfmin(Vu32.hf,Vv32.hf)`
///
/// Instruction Type: CVI_VX_LATE
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vfmin_hf))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vhf_vfmin_VhfVhf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vfmin_hf(vu, vv)
}

/// `Vd32.sf=vfmin(Vu32.sf,Vv32.sf)`
///
/// Instruction Type: CVI_VX_LATE
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vfmin_sf))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vsf_vfmin_VsfVsf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vfmin_sf(vu, vv)
}

/// `Vd32.hf=vfneg(Vu32.hf)`
///
/// Instruction Type: CVI_VX_LATE
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vfneg_hf))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vhf_vfneg_Vhf(vu: HvxVector) -> HvxVector {
    vfneg_hf(vu)
}

/// `Vd32.sf=vfneg(Vu32.sf)`
///
/// Instruction Type: CVI_VX_LATE
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vfneg_sf))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vsf_vfneg_Vsf(vu: HvxVector) -> HvxVector {
    vfneg_sf(vu)
}

/// `Vd32.hf=vmax(Vu32.hf,Vv32.hf)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vmax_hf))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vhf_vmax_VhfVhf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmax_hf(vu, vv)
}

/// `Vd32.sf=vmax(Vu32.sf,Vv32.sf)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vmax_sf))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vsf_vmax_VsfVsf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmax_sf(vu, vv)
}

/// `Vd32.hf=vmin(Vu32.hf,Vv32.hf)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vmin_hf))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vhf_vmin_VhfVhf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmin_hf(vu, vv)
}

/// `Vd32.sf=vmin(Vu32.sf,Vv32.sf)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vmin_sf))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vsf_vmin_VsfVsf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmin_sf(vu, vv)
}

/// `Vd32.hf=vmpy(Vu32.hf,Vv32.hf)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vmpy_hf_hf))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vhf_vmpy_VhfVhf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmpy_hf_hf(vu, vv)
}

/// `Vx32.hf+=vmpy(Vu32.hf,Vv32.hf)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vmpy_hf_hf_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vhf_vmpyacc_VhfVhfVhf(vx: HvxVector, vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmpy_hf_hf_acc(vx, vu, vv)
}

/// `Vd32.qf16=vmpy(Vu32.qf16,Vv32.qf16)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vmpy_qf16))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vqf16_vmpy_Vqf16Vqf16(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmpy_qf16(vu, vv)
}

/// `Vd32.qf16=vmpy(Vu32.hf,Vv32.hf)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vmpy_qf16_hf))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vqf16_vmpy_VhfVhf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmpy_qf16_hf(vu, vv)
}

/// `Vd32.qf16=vmpy(Vu32.qf16,Vv32.hf)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vmpy_qf16_mix_hf))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vqf16_vmpy_Vqf16Vhf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmpy_qf16_mix_hf(vu, vv)
}

/// `Vd32.qf32=vmpy(Vu32.qf32,Vv32.qf32)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vmpy_qf32))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vqf32_vmpy_Vqf32Vqf32(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmpy_qf32(vu, vv)
}

/// `Vdd32.qf32=vmpy(Vu32.hf,Vv32.hf)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vmpy_qf32_hf))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wqf32_vmpy_VhfVhf(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
    vmpy_qf32_hf(vu, vv)
}

/// `Vdd32.qf32=vmpy(Vu32.qf16,Vv32.hf)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vmpy_qf32_mix_hf))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wqf32_vmpy_Vqf16Vhf(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
    vmpy_qf32_mix_hf(vu, vv)
}

/// `Vdd32.qf32=vmpy(Vu32.qf16,Vv32.qf16)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vmpy_qf32_qf16))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wqf32_vmpy_Vqf16Vqf16(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
    vmpy_qf32_qf16(vu, vv)
}

/// `Vd32.qf32=vmpy(Vu32.sf,Vv32.sf)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vmpy_qf32_sf))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vqf32_vmpy_VsfVsf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmpy_qf32_sf(vu, vv)
}

/// `Vdd32.sf=vmpy(Vu32.hf,Vv32.hf)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vmpy_sf_hf))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wsf_vmpy_VhfVhf(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
    vmpy_sf_hf(vu, vv)
}

/// `Vxx32.sf+=vmpy(Vu32.hf,Vv32.hf)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vmpy_sf_hf_acc))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wsf_vmpyacc_WsfVhfVhf(
    vxx: HvxVectorPair,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPair {
    vmpy_sf_hf_acc(vxx, vu, vv)
}

/// `Vd32.sf=vmpy(Vu32.sf,Vv32.sf)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vmpy_sf_sf))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vsf_vmpy_VsfVsf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmpy_sf_sf(vu, vv)
}

/// `Vd32.qf16=vsub(Vu32.hf,Vv32.hf)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vsub_hf))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vqf16_vsub_VhfVhf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vsub_hf(vu, vv)
}

/// `Vd32.hf=vsub(Vu32.hf,Vv32.hf)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vsub_hf_hf))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vhf_vsub_VhfVhf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vsub_hf_hf(vu, vv)
}

/// `Vd32.qf16=vsub(Vu32.qf16,Vv32.qf16)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vsub_qf16))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vqf16_vsub_Vqf16Vqf16(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vsub_qf16(vu, vv)
}

/// `Vd32.qf16=vsub(Vu32.qf16,Vv32.hf)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vsub_qf16_mix))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vqf16_vsub_Vqf16Vhf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vsub_qf16_mix(vu, vv)
}

/// `Vd32.qf32=vsub(Vu32.qf32,Vv32.qf32)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vsub_qf32))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vqf32_vsub_Vqf32Vqf32(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vsub_qf32(vu, vv)
}

/// `Vd32.qf32=vsub(Vu32.qf32,Vv32.sf)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vsub_qf32_mix))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vqf32_vsub_Vqf32Vsf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vsub_qf32_mix(vu, vv)
}

/// `Vd32.qf32=vsub(Vu32.sf,Vv32.sf)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vsub_sf))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vqf32_vsub_VsfVsf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vsub_sf(vu, vv)
}

/// `Vdd32.sf=vsub(Vu32.hf,Vv32.hf)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vsub_sf_hf))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Wsf_vsub_VhfVhf(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
    vsub_sf_hf(vu, vv)
}

/// `Vd32.sf=vsub(Vu32.sf,Vv32.sf)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[cfg_attr(test, assert_instr(vsub_sf_sf))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vsf_vsub_VsfVsf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vsub_sf_sf(vu, vv)
}

/// `Vd32.ub=vasr(Vuu32.uh,Vv32.ub):rnd:sat`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv69"))]
#[cfg_attr(test, assert_instr(vasrvuhubrndsat))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vub_vasr_WuhVub_rnd_sat(vuu: HvxVectorPair, vv: HvxVector) -> HvxVector {
    vasrvuhubrndsat(vuu, vv)
}

/// `Vd32.ub=vasr(Vuu32.uh,Vv32.ub):sat`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv69"))]
#[cfg_attr(test, assert_instr(vasrvuhubsat))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vub_vasr_WuhVub_sat(vuu: HvxVectorPair, vv: HvxVector) -> HvxVector {
    vasrvuhubsat(vuu, vv)
}

/// `Vd32.uh=vasr(Vuu32.w,Vv32.uh):rnd:sat`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv69"))]
#[cfg_attr(test, assert_instr(vasrvwuhrndsat))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vuh_vasr_WwVuh_rnd_sat(vuu: HvxVectorPair, vv: HvxVector) -> HvxVector {
    vasrvwuhrndsat(vuu, vv)
}

/// `Vd32.uh=vasr(Vuu32.w,Vv32.uh):sat`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv69"))]
#[cfg_attr(test, assert_instr(vasrvwuhsat))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vuh_vasr_WwVuh_sat(vuu: HvxVectorPair, vv: HvxVector) -> HvxVector {
    vasrvwuhsat(vuu, vv)
}

/// `Vd32.uh=vmpy(Vu32.uh,Vv32.uh):>>16`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv69"))]
#[cfg_attr(test, assert_instr(vmpyuhvs))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vuh_vmpy_VuhVuh_rs16(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmpyuhvs(vu, vv)
}

/// `Vd32.h=Vu32.hf`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv73"))]
#[cfg_attr(test, assert_instr(vconv_h_hf))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vh_equals_Vhf(vu: HvxVector) -> HvxVector {
    vconv_h_hf(vu)
}

/// `Vd32.hf=Vu32.h`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv73"))]
#[cfg_attr(test, assert_instr(vconv_hf_h))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vhf_equals_Vh(vu: HvxVector) -> HvxVector {
    vconv_hf_h(vu)
}

/// `Vd32.sf=Vu32.w`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv73"))]
#[cfg_attr(test, assert_instr(vconv_sf_w))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vsf_equals_Vw(vu: HvxVector) -> HvxVector {
    vconv_sf_w(vu)
}

/// `Vd32.w=Vu32.sf`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv73"))]
#[cfg_attr(test, assert_instr(vconv_w_sf))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_equals_Vsf(vu: HvxVector) -> HvxVector {
    vconv_w_sf(vu)
}

/// `Vd32=vgetqfext(Vu32.x,Rt32)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv79"))]
#[cfg_attr(test, assert_instr(get_qfext))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_V_vgetqfext_VR(vu: HvxVector, rt: i32) -> HvxVector {
    get_qfext(vu, rt)
}

/// `Vd32.x=vsetqfext(Vu32,Rt32)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv79"))]
#[cfg_attr(test, assert_instr(set_qfext))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_V_vsetqfext_VR(vu: HvxVector, rt: i32) -> HvxVector {
    set_qfext(vu, rt)
}

/// `Vd32.f8=vabs(Vu32.f8)`
///
/// Instruction Type: CVI_VX_LATE
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv79"))]
#[cfg_attr(test, assert_instr(vabs_f8))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_V_vabs_V(vu: HvxVector) -> HvxVector {
    vabs_f8(vu)
}

/// `Vdd32.hf=vcvt2(Vu32.b)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv79"))]
#[cfg_attr(test, assert_instr(vcvt2_hf_b))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Whf_vcvt2_Vb(vu: HvxVector) -> HvxVectorPair {
    vcvt2_hf_b(vu)
}

/// `Vdd32.hf=vcvt2(Vu32.ub)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv79"))]
#[cfg_attr(test, assert_instr(vcvt2_hf_ub))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Whf_vcvt2_Vub(vu: HvxVector) -> HvxVectorPair {
    vcvt2_hf_ub(vu)
}

/// `Vdd32.hf=vcvt(Vu32.f8)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv79"))]
#[cfg_attr(test, assert_instr(vcvt_hf_f8))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Whf_vcvt_V(vu: HvxVector) -> HvxVectorPair {
    vcvt_hf_f8(vu)
}

/// `Vd32.f8=vfmax(Vu32.f8,Vv32.f8)`
///
/// Instruction Type: CVI_VX_LATE
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv79"))]
#[cfg_attr(test, assert_instr(vfmax_f8))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_V_vfmax_VV(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vfmax_f8(vu, vv)
}

/// `Vd32.f8=vfmin(Vu32.f8,Vv32.f8)`
///
/// Instruction Type: CVI_VX_LATE
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv79"))]
#[cfg_attr(test, assert_instr(vfmin_f8))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_V_vfmin_VV(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vfmin_f8(vu, vv)
}

/// `Vd32.f8=vfneg(Vu32.f8)`
///
/// Instruction Type: CVI_VX_LATE
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv79"))]
#[cfg_attr(test, assert_instr(vfneg_f8))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_V_vfneg_V(vu: HvxVector) -> HvxVector {
    vfneg_f8(vu)
}

/// `Qd4=and(Qs4,Qt4)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_and_QQ(qs: HvxVectorPred, qt: HvxVectorPred) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(
        pred_and(
            vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qs), -1),
            vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qt), -1),
        ),
        -1,
    ))
}

/// `Qd4=and(Qs4,!Qt4)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_and_QQn(qs: HvxVectorPred, qt: HvxVectorPred) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(
        pred_and_n(
            vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qs), -1),
            vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qt), -1),
        ),
        -1,
    ))
}

/// `Qd4=not(Qs4)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_not_Q(qs: HvxVectorPred) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(
        pred_not(vandvrt(
            core::mem::transmute::<HvxVectorPred, HvxVector>(qs),
            -1,
        )),
        -1,
    ))
}

/// `Qd4=or(Qs4,Qt4)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_or_QQ(qs: HvxVectorPred, qt: HvxVectorPred) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(
        pred_or(
            vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qs), -1),
            vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qt), -1),
        ),
        -1,
    ))
}

/// `Qd4=or(Qs4,!Qt4)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_or_QQn(qs: HvxVectorPred, qt: HvxVectorPred) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(
        pred_or_n(
            vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qs), -1),
            vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qt), -1),
        ),
        -1,
    ))
}

/// `Qd4=vsetq(Rt32)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VP
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_vsetq_R(rt: i32) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(pred_scalar2(rt), -1))
}

/// `Qd4=xor(Qs4,Qt4)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_xor_QQ(qs: HvxVectorPred, qt: HvxVectorPred) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(
        pred_xor(
            vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qs), -1),
            vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qt), -1),
        ),
        -1,
    ))
}

/// `if (!Qv4) vmem(Rt32+#s4)=Vs32`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VM_ST
/// Execution Slots: SLOT0
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_vmem_QnRIV(qv: HvxVectorPred, rt: *mut HvxVector, vs: HvxVector) {
    vS32b_nqpred_ai(
        vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qv), -1),
        rt,
        vs,
    )
}

/// `if (!Qv4) vmem(Rt32+#s4):nt=Vs32`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VM_ST
/// Execution Slots: SLOT0
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_vmem_QnRIV_nt(qv: HvxVectorPred, rt: *mut HvxVector, vs: HvxVector) {
    vS32b_nt_nqpred_ai(
        vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qv), -1),
        rt,
        vs,
    )
}

/// `if (Qv4) vmem(Rt32+#s4):nt=Vs32`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VM_ST
/// Execution Slots: SLOT0
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_vmem_QRIV_nt(qv: HvxVectorPred, rt: *mut HvxVector, vs: HvxVector) {
    vS32b_nt_qpred_ai(
        vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qv), -1),
        rt,
        vs,
    )
}

/// `if (Qv4) vmem(Rt32+#s4)=Vs32`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VM_ST
/// Execution Slots: SLOT0
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_vmem_QRIV(qv: HvxVectorPred, rt: *mut HvxVector, vs: HvxVector) {
    vS32b_qpred_ai(
        vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qv), -1),
        rt,
        vs,
    )
}

/// `if (!Qv4) Vx32.b+=Vu32.b`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vb_condacc_QnVbVb(qv: HvxVectorPred, vx: HvxVector, vu: HvxVector) -> HvxVector {
    vaddbnq(
        vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qv), -1),
        vx,
        vu,
    )
}

/// `if (Qv4) Vx32.b+=Vu32.b`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vb_condacc_QVbVb(qv: HvxVectorPred, vx: HvxVector, vu: HvxVector) -> HvxVector {
    vaddbq(
        vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qv), -1),
        vx,
        vu,
    )
}

/// `if (!Qv4) Vx32.h+=Vu32.h`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vh_condacc_QnVhVh(qv: HvxVectorPred, vx: HvxVector, vu: HvxVector) -> HvxVector {
    vaddhnq(
        vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qv), -1),
        vx,
        vu,
    )
}

/// `if (Qv4) Vx32.h+=Vu32.h`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vh_condacc_QVhVh(qv: HvxVectorPred, vx: HvxVector, vu: HvxVector) -> HvxVector {
    vaddhq(
        vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qv), -1),
        vx,
        vu,
    )
}

/// `if (!Qv4) Vx32.w+=Vu32.w`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_condacc_QnVwVw(qv: HvxVectorPred, vx: HvxVector, vu: HvxVector) -> HvxVector {
    vaddwnq(
        vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qv), -1),
        vx,
        vu,
    )
}

/// `if (Qv4) Vx32.w+=Vu32.w`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_condacc_QVwVw(qv: HvxVectorPred, vx: HvxVector, vu: HvxVector) -> HvxVector {
    vaddwq(
        vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qv), -1),
        vx,
        vu,
    )
}

/// `Vd32=vand(Qu4,Rt32)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VX_LATE
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_V_vand_QR(qu: HvxVectorPred, rt: i32) -> HvxVector {
    vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qu), rt)
}

/// `Vx32|=vand(Qu4,Rt32)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VX_LATE
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_V_vandor_VQR(vx: HvxVector, qu: HvxVectorPred, rt: i32) -> HvxVector {
    vandvrt_acc(vx, core::mem::transmute::<HvxVectorPred, HvxVector>(qu), rt)
}

/// `Qd4=vand(Vu32,Rt32)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VX_LATE
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_vand_VR(vu: HvxVector, rt: i32) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(vu, rt))
}

/// `Qx4|=vand(Vu32,Rt32)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VX_LATE
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_vandor_QVR(qx: HvxVectorPred, vu: HvxVector, rt: i32) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt_acc(
        core::mem::transmute::<HvxVectorPred, HvxVector>(qx),
        vu,
        rt,
    ))
}

/// `Qd4=vcmp.eq(Vu32.b,Vv32.b)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_vcmp_eq_VbVb(vu: HvxVector, vv: HvxVector) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(veqb(vu, vv), -1))
}

/// `Qx4&=vcmp.eq(Vu32.b,Vv32.b)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_vcmp_eqand_QVbVb(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(
        veqb_and(
            vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qx), -1),
            vu,
            vv,
        ),
        -1,
    ))
}

/// `Qx4|=vcmp.eq(Vu32.b,Vv32.b)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_vcmp_eqor_QVbVb(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(
        veqb_or(
            vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qx), -1),
            vu,
            vv,
        ),
        -1,
    ))
}

/// `Qx4^=vcmp.eq(Vu32.b,Vv32.b)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_vcmp_eqxacc_QVbVb(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(
        veqb_xor(
            vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qx), -1),
            vu,
            vv,
        ),
        -1,
    ))
}

/// `Qd4=vcmp.eq(Vu32.h,Vv32.h)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_vcmp_eq_VhVh(vu: HvxVector, vv: HvxVector) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(veqh(vu, vv), -1))
}

/// `Qx4&=vcmp.eq(Vu32.h,Vv32.h)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_vcmp_eqand_QVhVh(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(
        veqh_and(
            vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qx), -1),
            vu,
            vv,
        ),
        -1,
    ))
}

/// `Qx4|=vcmp.eq(Vu32.h,Vv32.h)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_vcmp_eqor_QVhVh(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(
        veqh_or(
            vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qx), -1),
            vu,
            vv,
        ),
        -1,
    ))
}

/// `Qx4^=vcmp.eq(Vu32.h,Vv32.h)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_vcmp_eqxacc_QVhVh(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(
        veqh_xor(
            vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qx), -1),
            vu,
            vv,
        ),
        -1,
    ))
}

/// `Qd4=vcmp.eq(Vu32.w,Vv32.w)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_vcmp_eq_VwVw(vu: HvxVector, vv: HvxVector) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(veqw(vu, vv), -1))
}

/// `Qx4&=vcmp.eq(Vu32.w,Vv32.w)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_vcmp_eqand_QVwVw(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(
        veqw_and(
            vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qx), -1),
            vu,
            vv,
        ),
        -1,
    ))
}

/// `Qx4|=vcmp.eq(Vu32.w,Vv32.w)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_vcmp_eqor_QVwVw(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(
        veqw_or(
            vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qx), -1),
            vu,
            vv,
        ),
        -1,
    ))
}

/// `Qx4^=vcmp.eq(Vu32.w,Vv32.w)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_vcmp_eqxacc_QVwVw(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(
        veqw_xor(
            vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qx), -1),
            vu,
            vv,
        ),
        -1,
    ))
}

/// `Qd4=vcmp.gt(Vu32.b,Vv32.b)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_vcmp_gt_VbVb(vu: HvxVector, vv: HvxVector) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(vgtb(vu, vv), -1))
}

/// `Qx4&=vcmp.gt(Vu32.b,Vv32.b)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_vcmp_gtand_QVbVb(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(
        vgtb_and(
            vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qx), -1),
            vu,
            vv,
        ),
        -1,
    ))
}

/// `Qx4|=vcmp.gt(Vu32.b,Vv32.b)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_vcmp_gtor_QVbVb(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(
        vgtb_or(
            vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qx), -1),
            vu,
            vv,
        ),
        -1,
    ))
}

/// `Qx4^=vcmp.gt(Vu32.b,Vv32.b)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_vcmp_gtxacc_QVbVb(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(
        vgtb_xor(
            vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qx), -1),
            vu,
            vv,
        ),
        -1,
    ))
}

/// `Qd4=vcmp.gt(Vu32.h,Vv32.h)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_vcmp_gt_VhVh(vu: HvxVector, vv: HvxVector) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(vgth(vu, vv), -1))
}

/// `Qx4&=vcmp.gt(Vu32.h,Vv32.h)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_vcmp_gtand_QVhVh(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(
        vgth_and(
            vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qx), -1),
            vu,
            vv,
        ),
        -1,
    ))
}

/// `Qx4|=vcmp.gt(Vu32.h,Vv32.h)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_vcmp_gtor_QVhVh(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(
        vgth_or(
            vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qx), -1),
            vu,
            vv,
        ),
        -1,
    ))
}

/// `Qx4^=vcmp.gt(Vu32.h,Vv32.h)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_vcmp_gtxacc_QVhVh(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(
        vgth_xor(
            vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qx), -1),
            vu,
            vv,
        ),
        -1,
    ))
}

/// `Qd4=vcmp.gt(Vu32.ub,Vv32.ub)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_vcmp_gt_VubVub(vu: HvxVector, vv: HvxVector) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(vgtub(vu, vv), -1))
}

/// `Qx4&=vcmp.gt(Vu32.ub,Vv32.ub)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_vcmp_gtand_QVubVub(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(
        vgtub_and(
            vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qx), -1),
            vu,
            vv,
        ),
        -1,
    ))
}

/// `Qx4|=vcmp.gt(Vu32.ub,Vv32.ub)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_vcmp_gtor_QVubVub(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(
        vgtub_or(
            vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qx), -1),
            vu,
            vv,
        ),
        -1,
    ))
}

/// `Qx4^=vcmp.gt(Vu32.ub,Vv32.ub)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_vcmp_gtxacc_QVubVub(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(
        vgtub_xor(
            vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qx), -1),
            vu,
            vv,
        ),
        -1,
    ))
}

/// `Qd4=vcmp.gt(Vu32.uh,Vv32.uh)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_vcmp_gt_VuhVuh(vu: HvxVector, vv: HvxVector) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(vgtuh(vu, vv), -1))
}

/// `Qx4&=vcmp.gt(Vu32.uh,Vv32.uh)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_vcmp_gtand_QVuhVuh(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(
        vgtuh_and(
            vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qx), -1),
            vu,
            vv,
        ),
        -1,
    ))
}

/// `Qx4|=vcmp.gt(Vu32.uh,Vv32.uh)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_vcmp_gtor_QVuhVuh(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(
        vgtuh_or(
            vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qx), -1),
            vu,
            vv,
        ),
        -1,
    ))
}

/// `Qx4^=vcmp.gt(Vu32.uh,Vv32.uh)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_vcmp_gtxacc_QVuhVuh(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(
        vgtuh_xor(
            vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qx), -1),
            vu,
            vv,
        ),
        -1,
    ))
}

/// `Qd4=vcmp.gt(Vu32.uw,Vv32.uw)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_vcmp_gt_VuwVuw(vu: HvxVector, vv: HvxVector) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(vgtuw(vu, vv), -1))
}

/// `Qx4&=vcmp.gt(Vu32.uw,Vv32.uw)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_vcmp_gtand_QVuwVuw(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(
        vgtuw_and(
            vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qx), -1),
            vu,
            vv,
        ),
        -1,
    ))
}

/// `Qx4|=vcmp.gt(Vu32.uw,Vv32.uw)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_vcmp_gtor_QVuwVuw(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(
        vgtuw_or(
            vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qx), -1),
            vu,
            vv,
        ),
        -1,
    ))
}

/// `Qx4^=vcmp.gt(Vu32.uw,Vv32.uw)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_vcmp_gtxacc_QVuwVuw(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(
        vgtuw_xor(
            vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qx), -1),
            vu,
            vv,
        ),
        -1,
    ))
}

/// `Qd4=vcmp.gt(Vu32.w,Vv32.w)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_vcmp_gt_VwVw(vu: HvxVector, vv: HvxVector) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(vgtw(vu, vv), -1))
}

/// `Qx4&=vcmp.gt(Vu32.w,Vv32.w)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_vcmp_gtand_QVwVw(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(
        vgtw_and(
            vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qx), -1),
            vu,
            vv,
        ),
        -1,
    ))
}

/// `Qx4|=vcmp.gt(Vu32.w,Vv32.w)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_vcmp_gtor_QVwVw(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(
        vgtw_or(
            vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qx), -1),
            vu,
            vv,
        ),
        -1,
    ))
}

/// `Qx4^=vcmp.gt(Vu32.w,Vv32.w)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_vcmp_gtxacc_QVwVw(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(
        vgtw_xor(
            vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qx), -1),
            vu,
            vv,
        ),
        -1,
    ))
}

/// `Vd32=vmux(Qt4,Vu32,Vv32)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_V_vmux_QVV(qt: HvxVectorPred, vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmux(
        vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qt), -1),
        vu,
        vv,
    )
}

/// `if (!Qv4) Vx32.b-=Vu32.b`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vb_condnac_QnVbVb(qv: HvxVectorPred, vx: HvxVector, vu: HvxVector) -> HvxVector {
    vsubbnq(
        vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qv), -1),
        vx,
        vu,
    )
}

/// `if (Qv4) Vx32.b-=Vu32.b`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vb_condnac_QVbVb(qv: HvxVectorPred, vx: HvxVector, vu: HvxVector) -> HvxVector {
    vsubbq(
        vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qv), -1),
        vx,
        vu,
    )
}

/// `if (!Qv4) Vx32.h-=Vu32.h`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vh_condnac_QnVhVh(qv: HvxVectorPred, vx: HvxVector, vu: HvxVector) -> HvxVector {
    vsubhnq(
        vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qv), -1),
        vx,
        vu,
    )
}

/// `if (Qv4) Vx32.h-=Vu32.h`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vh_condnac_QVhVh(qv: HvxVectorPred, vx: HvxVector, vu: HvxVector) -> HvxVector {
    vsubhq(
        vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qv), -1),
        vx,
        vu,
    )
}

/// `if (!Qv4) Vx32.w-=Vu32.w`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_condnac_QnVwVw(qv: HvxVectorPred, vx: HvxVector, vu: HvxVector) -> HvxVector {
    vsubwnq(
        vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qv), -1),
        vx,
        vu,
    )
}

/// `if (Qv4) Vx32.w-=Vu32.w`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_condnac_QVwVw(qv: HvxVectorPred, vx: HvxVector, vu: HvxVector) -> HvxVector {
    vsubwq(
        vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qv), -1),
        vx,
        vu,
    )
}

/// `Vdd32=vswap(Qt4,Vu32,Vv32)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_W_vswap_QVV(qt: HvxVectorPred, vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
    vswap(
        vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qt), -1),
        vu,
        vv,
    )
}

/// `Qd4=vsetq2(Rt32)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VP
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_vsetq2_R(rt: i32) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(pred_scalar2v2(rt), -1))
}

/// `Qd4.b=vshuffe(Qs4.h,Qt4.h)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Qb_vshuffe_QhQh(qs: HvxVectorPred, qt: HvxVectorPred) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(
        shuffeqh(
            vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qs), -1),
            vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qt), -1),
        ),
        -1,
    ))
}

/// `Qd4.h=vshuffe(Qs4.w,Qt4.w)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Qh_vshuffe_QwQw(qs: HvxVectorPred, qt: HvxVectorPred) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(
        shuffeqw(
            vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qs), -1),
            vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qt), -1),
        ),
        -1,
    ))
}

/// `Vd32=vand(!Qu4,Rt32)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VX_LATE
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_V_vand_QnR(qu: HvxVectorPred, rt: i32) -> HvxVector {
    vandnqrt(
        vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qu), -1),
        rt,
    )
}

/// `Vx32|=vand(!Qu4,Rt32)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VX_LATE
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_V_vandor_VQnR(vx: HvxVector, qu: HvxVectorPred, rt: i32) -> HvxVector {
    vandnqrt_acc(
        vx,
        vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qu), -1),
        rt,
    )
}

/// `Vd32=vand(!Qv4,Vu32)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_V_vand_QnV(qv: HvxVectorPred, vu: HvxVector) -> HvxVector {
    vandvnqv(
        vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qv), -1),
        vu,
    )
}

/// `Vd32=vand(Qv4,Vu32)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_V_vand_QV(qv: HvxVectorPred, vu: HvxVector) -> HvxVector {
    vandvqv(
        vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qv), -1),
        vu,
    )
}

/// `if (Qs4) vtmp.h=vgather(Rt32,Mu2,Vv32.h).h`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_GATHER
/// Execution Slots: SLOT01
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_vgather_AQRMVh(
    rs: *mut HvxVector,
    qs: HvxVectorPred,
    rt: i32,
    mu: i32,
    vv: HvxVector,
) {
    vgathermhq(
        rs,
        vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qs), -1),
        rt,
        mu,
        vv,
    )
}

/// `if (Qs4) vtmp.h=vgather(Rt32,Mu2,Vvv32.w).h`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_GATHER_DV
/// Execution Slots: SLOT01
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_vgather_AQRMWw(
    rs: *mut HvxVector,
    qs: HvxVectorPred,
    rt: i32,
    mu: i32,
    vvv: HvxVectorPair,
) {
    vgathermhwq(
        rs,
        vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qs), -1),
        rt,
        mu,
        vvv,
    )
}

/// `if (Qs4) vtmp.w=vgather(Rt32,Mu2,Vv32.w).w`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_GATHER
/// Execution Slots: SLOT01
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_vgather_AQRMVw(
    rs: *mut HvxVector,
    qs: HvxVectorPred,
    rt: i32,
    mu: i32,
    vv: HvxVector,
) {
    vgathermwq(
        rs,
        vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qs), -1),
        rt,
        mu,
        vv,
    )
}

/// `Vd32.b=prefixsum(Qv4)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vb_prefixsum_Q(qv: HvxVectorPred) -> HvxVector {
    vprefixqb(vandvrt(
        core::mem::transmute::<HvxVectorPred, HvxVector>(qv),
        -1,
    ))
}

/// `Vd32.h=prefixsum(Qv4)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vh_prefixsum_Q(qv: HvxVectorPred) -> HvxVector {
    vprefixqh(vandvrt(
        core::mem::transmute::<HvxVectorPred, HvxVector>(qv),
        -1,
    ))
}

/// `Vd32.w=prefixsum(Qv4)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_prefixsum_Q(qv: HvxVectorPred) -> HvxVector {
    vprefixqw(vandvrt(
        core::mem::transmute::<HvxVectorPred, HvxVector>(qv),
        -1,
    ))
}

/// `if (Qs4) vscatter(Rt32,Mu2,Vv32.h).h=Vw32`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_SCATTER
/// Execution Slots: SLOT0
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_vscatter_QRMVhV(
    qs: HvxVectorPred,
    rt: i32,
    mu: i32,
    vv: HvxVector,
    vw: HvxVector,
) {
    vscattermhq(
        vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qs), -1),
        rt,
        mu,
        vv,
        vw,
    )
}

/// `if (Qs4) vscatter(Rt32,Mu2,Vvv32.w).h=Vw32`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_SCATTER_DV
/// Execution Slots: SLOT0
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_vscatter_QRMWwV(
    qs: HvxVectorPred,
    rt: i32,
    mu: i32,
    vvv: HvxVectorPair,
    vw: HvxVector,
) {
    vscattermhwq(
        vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qs), -1),
        rt,
        mu,
        vvv,
        vw,
    )
}

/// `if (Qs4) vscatter(Rt32,Mu2,Vv32.w).w=Vw32`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_SCATTER
/// Execution Slots: SLOT0
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_vscatter_QRMVwV(
    qs: HvxVectorPred,
    rt: i32,
    mu: i32,
    vv: HvxVector,
    vw: HvxVector,
) {
    vscattermwq(
        vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qs), -1),
        rt,
        mu,
        vv,
        vw,
    )
}

/// `Vd32.w=vadd(Vu32.w,Vv32.w,Qs4):carry:sat`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv66"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Vw_vadd_VwVwQ_carry_sat(
    vu: HvxVector,
    vv: HvxVector,
    qs: HvxVectorPred,
) -> HvxVector {
    vaddcarrysat(
        vu,
        vv,
        vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qs), -1),
    )
}

/// `Qd4=vcmp.gt(Vu32.hf,Vv32.hf)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_vcmp_gt_VhfVhf(vu: HvxVector, vv: HvxVector) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(vgthf(vu, vv), -1))
}

/// `Qx4&=vcmp.gt(Vu32.hf,Vv32.hf)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_vcmp_gtand_QVhfVhf(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(
        vgthf_and(
            vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qx), -1),
            vu,
            vv,
        ),
        -1,
    ))
}

/// `Qx4|=vcmp.gt(Vu32.hf,Vv32.hf)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_vcmp_gtor_QVhfVhf(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(
        vgthf_or(
            vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qx), -1),
            vu,
            vv,
        ),
        -1,
    ))
}

/// `Qx4^=vcmp.gt(Vu32.hf,Vv32.hf)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_vcmp_gtxacc_QVhfVhf(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(
        vgthf_xor(
            vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qx), -1),
            vu,
            vv,
        ),
        -1,
    ))
}

/// `Qd4=vcmp.gt(Vu32.sf,Vv32.sf)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_vcmp_gt_VsfVsf(vu: HvxVector, vv: HvxVector) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(vgtsf(vu, vv), -1))
}

/// `Qx4&=vcmp.gt(Vu32.sf,Vv32.sf)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_vcmp_gtand_QVsfVsf(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(
        vgtsf_and(
            vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qx), -1),
            vu,
            vv,
        ),
        -1,
    ))
}

/// `Qx4|=vcmp.gt(Vu32.sf,Vv32.sf)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_vcmp_gtor_QVsfVsf(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(
        vgtsf_or(
            vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qx), -1),
            vu,
            vv,
        ),
        -1,
    ))
}

/// `Qx4^=vcmp.gt(Vu32.sf,Vv32.sf)`
///
/// This is a compound operation composed of multiple HVX instructions.
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Q_vcmp_gtxacc_QVsfVsf(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    core::mem::transmute::<HvxVector, HvxVectorPred>(vandqrt(
        vgtsf_xor(
            vandvrt(core::mem::transmute::<HvxVectorPred, HvxVector>(qx), -1),
            vu,
            vv,
        ),
        -1,
    ))
}
