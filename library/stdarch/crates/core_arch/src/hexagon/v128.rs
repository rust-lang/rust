//! Hexagon HVX 128-byte vector mode intrinsics
//!
//! This module provides intrinsics for the Hexagon Vector Extensions (HVX)
//! in 128-byte vector mode (1024-bit vectors).
//!
//! HVX is a wide vector extension designed for high-performance signal processing.
//! [Hexagon HVX Programmer's Reference Manual](https://docs.qualcomm.com/doc/80-N2040-61)
//!
//! ## Vector Types
//!
//! In 128-byte mode:
//! - `HvxVector` is 1024 bits (128 bytes) containing 32 x 32-bit values
//! - `HvxVectorPair` is 2048 bits (256 bytes)
//! - `HvxVectorPred` is 1024 bits (128 bytes) for predicate operations
//!
//! To use this module, compile with `-C target-feature=+hvx-length128b`.
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

#[cfg(test)]
use stdarch_test::assert_instr;

use crate::intrinsics::simd::{simd_add, simd_and, simd_or, simd_sub, simd_xor};

// HVX type definitions for 128-byte vector mode
types! {
    #![unstable(feature = "stdarch_hexagon", issue = "151523")]

    /// HVX vector type (1024 bits / 128 bytes)
    ///
    /// This type represents a single HVX vector register containing 32 x 32-bit values.
    pub struct HvxVector(32 x i32);

    /// HVX vector pair type (2048 bits / 256 bytes)
    ///
    /// This type represents a pair of HVX vector registers, often used for
    /// operations that produce double-width results.
    pub struct HvxVectorPair(64 x i32);

    /// HVX vector predicate type (1024 bits / 128 bytes)
    ///
    /// This type represents a predicate vector used for conditional operations.
    /// Each bit corresponds to a lane in the vector.
    pub struct HvxVectorPred(32 x i32);
}

// LLVM intrinsic declarations for 128-byte vector mode
#[allow(improper_ctypes)]
unsafe extern "unadjusted" {
    #[link_name = "llvm.hexagon.V6.extractw.128B"]
    fn extractw(_: HvxVector, _: i32) -> i32;
    #[link_name = "llvm.hexagon.V6.get.qfext.128B"]
    fn get_qfext(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.hi.128B"]
    fn hi(_: HvxVectorPair) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.lo.128B"]
    fn lo(_: HvxVectorPair) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.lvsplatb.128B"]
    fn lvsplatb(_: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.lvsplath.128B"]
    fn lvsplath(_: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.lvsplatw.128B"]
    fn lvsplatw(_: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.pred.and.128B"]
    fn pred_and(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.pred.and.n.128B"]
    fn pred_and_n(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.pred.not.128B"]
    fn pred_not(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.pred.or.128B"]
    fn pred_or(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.pred.or.n.128B"]
    fn pred_or_n(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.pred.scalar2.128B"]
    fn pred_scalar2(_: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.pred.scalar2v2.128B"]
    fn pred_scalar2v2(_: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.pred.xor.128B"]
    fn pred_xor(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.set.qfext.128B"]
    fn set_qfext(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.shuffeqh.128B"]
    fn shuffeqh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.shuffeqw.128B"]
    fn shuffeqw(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.v6mpyhubs10.128B"]
    fn v6mpyhubs10(_: HvxVectorPair, _: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.v6mpyhubs10.vxx.128B"]
    fn v6mpyhubs10_vxx(
        _: HvxVectorPair,
        _: HvxVectorPair,
        _: HvxVectorPair,
        _: i32,
    ) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.v6mpyvubs10.128B"]
    fn v6mpyvubs10(_: HvxVectorPair, _: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.v6mpyvubs10.vxx.128B"]
    fn v6mpyvubs10_vxx(
        _: HvxVectorPair,
        _: HvxVectorPair,
        _: HvxVectorPair,
        _: i32,
    ) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vS32b.nqpred.ai.128B"]
    fn vS32b_nqpred_ai(_: HvxVector, _: *mut HvxVector, _: HvxVector) -> ();
    #[link_name = "llvm.hexagon.V6.vS32b.nt.nqpred.ai.128B"]
    fn vS32b_nt_nqpred_ai(_: HvxVector, _: *mut HvxVector, _: HvxVector) -> ();
    #[link_name = "llvm.hexagon.V6.vS32b.nt.qpred.ai.128B"]
    fn vS32b_nt_qpred_ai(_: HvxVector, _: *mut HvxVector, _: HvxVector) -> ();
    #[link_name = "llvm.hexagon.V6.vS32b.qpred.ai.128B"]
    fn vS32b_qpred_ai(_: HvxVector, _: *mut HvxVector, _: HvxVector) -> ();
    #[link_name = "llvm.hexagon.V6.vabs.f8.128B"]
    fn vabs_f8(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vabs.hf.128B"]
    fn vabs_hf(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vabs.sf.128B"]
    fn vabs_sf(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vabsb.128B"]
    fn vabsb(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vabsb.sat.128B"]
    fn vabsb_sat(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vabsdiffh.128B"]
    fn vabsdiffh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vabsdiffub.128B"]
    fn vabsdiffub(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vabsdiffuh.128B"]
    fn vabsdiffuh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vabsdiffw.128B"]
    fn vabsdiffw(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vabsh.128B"]
    fn vabsh(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vabsh.sat.128B"]
    fn vabsh_sat(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vabsw.128B"]
    fn vabsw(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vabsw.sat.128B"]
    fn vabsw_sat(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vadd.hf.128B"]
    fn vadd_hf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vadd.hf.hf.128B"]
    fn vadd_hf_hf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vadd.qf16.128B"]
    fn vadd_qf16(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vadd.qf16.mix.128B"]
    fn vadd_qf16_mix(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vadd.qf32.128B"]
    fn vadd_qf32(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vadd.qf32.mix.128B"]
    fn vadd_qf32_mix(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vadd.sf.128B"]
    fn vadd_sf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vadd.sf.hf.128B"]
    fn vadd_sf_hf(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vadd.sf.sf.128B"]
    fn vadd_sf_sf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaddb.128B"]
    fn vaddb(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaddb.dv.128B"]
    fn vaddb_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vaddbnq.128B"]
    fn vaddbnq(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaddbq.128B"]
    fn vaddbq(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaddbsat.128B"]
    fn vaddbsat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaddbsat.dv.128B"]
    fn vaddbsat_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vaddcarrysat.128B"]
    fn vaddcarrysat(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaddclbh.128B"]
    fn vaddclbh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaddclbw.128B"]
    fn vaddclbw(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaddh.128B"]
    fn vaddh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaddh.dv.128B"]
    fn vaddh_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vaddhnq.128B"]
    fn vaddhnq(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaddhq.128B"]
    fn vaddhq(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaddhsat.128B"]
    fn vaddhsat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaddhsat.dv.128B"]
    fn vaddhsat_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vaddhw.128B"]
    fn vaddhw(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vaddhw.acc.128B"]
    fn vaddhw_acc(_: HvxVectorPair, _: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vaddubh.128B"]
    fn vaddubh(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vaddubh.acc.128B"]
    fn vaddubh_acc(_: HvxVectorPair, _: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vaddubsat.128B"]
    fn vaddubsat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaddubsat.dv.128B"]
    fn vaddubsat_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vaddububb.sat.128B"]
    fn vaddububb_sat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vadduhsat.128B"]
    fn vadduhsat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vadduhsat.dv.128B"]
    fn vadduhsat_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vadduhw.128B"]
    fn vadduhw(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vadduhw.acc.128B"]
    fn vadduhw_acc(_: HvxVectorPair, _: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vadduwsat.128B"]
    fn vadduwsat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vadduwsat.dv.128B"]
    fn vadduwsat_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vaddw.128B"]
    fn vaddw(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaddw.dv.128B"]
    fn vaddw_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vaddwnq.128B"]
    fn vaddwnq(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaddwq.128B"]
    fn vaddwq(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaddwsat.128B"]
    fn vaddwsat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaddwsat.dv.128B"]
    fn vaddwsat_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.valignb.128B"]
    fn valignb(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.valignbi.128B"]
    fn valignbi(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vand.128B"]
    fn vand(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vandnqrt.128B"]
    fn vandnqrt(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vandnqrt.acc.128B"]
    fn vandnqrt_acc(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vandqrt.128B"]
    fn vandqrt(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vandqrt.acc.128B"]
    fn vandqrt_acc(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vandvnqv.128B"]
    fn vandvnqv(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vandvqv.128B"]
    fn vandvqv(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vandvrt.128B"]
    fn vandvrt(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vandvrt.acc.128B"]
    fn vandvrt_acc(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaslh.128B"]
    fn vaslh(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaslh.acc.128B"]
    fn vaslh_acc(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaslhv.128B"]
    fn vaslhv(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaslw.128B"]
    fn vaslw(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaslw.acc.128B"]
    fn vaslw_acc(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaslwv.128B"]
    fn vaslwv(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vasr.into.128B"]
    fn vasr_into(_: HvxVectorPair, _: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vasrh.128B"]
    fn vasrh(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vasrh.acc.128B"]
    fn vasrh_acc(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vasrhbrndsat.128B"]
    fn vasrhbrndsat(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vasrhbsat.128B"]
    fn vasrhbsat(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vasrhubrndsat.128B"]
    fn vasrhubrndsat(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vasrhubsat.128B"]
    fn vasrhubsat(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vasrhv.128B"]
    fn vasrhv(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vasruhubrndsat.128B"]
    fn vasruhubrndsat(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vasruhubsat.128B"]
    fn vasruhubsat(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vasruwuhrndsat.128B"]
    fn vasruwuhrndsat(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vasruwuhsat.128B"]
    fn vasruwuhsat(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vasrvuhubrndsat.128B"]
    fn vasrvuhubrndsat(_: HvxVectorPair, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vasrvuhubsat.128B"]
    fn vasrvuhubsat(_: HvxVectorPair, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vasrvwuhrndsat.128B"]
    fn vasrvwuhrndsat(_: HvxVectorPair, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vasrvwuhsat.128B"]
    fn vasrvwuhsat(_: HvxVectorPair, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vasrw.128B"]
    fn vasrw(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vasrw.acc.128B"]
    fn vasrw_acc(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vasrwh.128B"]
    fn vasrwh(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vasrwhrndsat.128B"]
    fn vasrwhrndsat(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vasrwhsat.128B"]
    fn vasrwhsat(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vasrwuhrndsat.128B"]
    fn vasrwuhrndsat(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vasrwuhsat.128B"]
    fn vasrwuhsat(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vasrwv.128B"]
    fn vasrwv(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vassign.128B"]
    fn vassign(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vassign.fp.128B"]
    fn vassign_fp(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vassignp.128B"]
    fn vassignp(_: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vavgb.128B"]
    fn vavgb(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vavgbrnd.128B"]
    fn vavgbrnd(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vavgh.128B"]
    fn vavgh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vavghrnd.128B"]
    fn vavghrnd(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vavgub.128B"]
    fn vavgub(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vavgubrnd.128B"]
    fn vavgubrnd(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vavguh.128B"]
    fn vavguh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vavguhrnd.128B"]
    fn vavguhrnd(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vavguw.128B"]
    fn vavguw(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vavguwrnd.128B"]
    fn vavguwrnd(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vavgw.128B"]
    fn vavgw(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vavgwrnd.128B"]
    fn vavgwrnd(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vcl0h.128B"]
    fn vcl0h(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vcl0w.128B"]
    fn vcl0w(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vcombine.128B"]
    fn vcombine(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vconv.h.hf.128B"]
    fn vconv_h_hf(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vconv.hf.h.128B"]
    fn vconv_hf_h(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vconv.hf.qf16.128B"]
    fn vconv_hf_qf16(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vconv.hf.qf32.128B"]
    fn vconv_hf_qf32(_: HvxVectorPair) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vconv.sf.qf32.128B"]
    fn vconv_sf_qf32(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vconv.sf.w.128B"]
    fn vconv_sf_w(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vconv.w.sf.128B"]
    fn vconv_w_sf(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vcvt2.hf.b.128B"]
    fn vcvt2_hf_b(_: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vcvt2.hf.ub.128B"]
    fn vcvt2_hf_ub(_: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vcvt.b.hf.128B"]
    fn vcvt_b_hf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vcvt.h.hf.128B"]
    fn vcvt_h_hf(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vcvt.hf.b.128B"]
    fn vcvt_hf_b(_: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vcvt.hf.f8.128B"]
    fn vcvt_hf_f8(_: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vcvt.hf.h.128B"]
    fn vcvt_hf_h(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vcvt.hf.sf.128B"]
    fn vcvt_hf_sf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vcvt.hf.ub.128B"]
    fn vcvt_hf_ub(_: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vcvt.hf.uh.128B"]
    fn vcvt_hf_uh(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vcvt.sf.hf.128B"]
    fn vcvt_sf_hf(_: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vcvt.ub.hf.128B"]
    fn vcvt_ub_hf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vcvt.uh.hf.128B"]
    fn vcvt_uh_hf(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vd0.128B"]
    fn vd0() -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdd0.128B"]
    fn vdd0() -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vdealb.128B"]
    fn vdealb(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdealb4w.128B"]
    fn vdealb4w(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdealh.128B"]
    fn vdealh(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdealvdd.128B"]
    fn vdealvdd(_: HvxVector, _: HvxVector, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vdelta.128B"]
    fn vdelta(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdmpy.sf.hf.128B"]
    fn vdmpy_sf_hf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdmpy.sf.hf.acc.128B"]
    fn vdmpy_sf_hf_acc(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdmpybus.128B"]
    fn vdmpybus(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdmpybus.acc.128B"]
    fn vdmpybus_acc(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdmpybus.dv.128B"]
    fn vdmpybus_dv(_: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vdmpybus.dv.acc.128B"]
    fn vdmpybus_dv_acc(_: HvxVectorPair, _: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vdmpyhb.128B"]
    fn vdmpyhb(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdmpyhb.acc.128B"]
    fn vdmpyhb_acc(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdmpyhb.dv.128B"]
    fn vdmpyhb_dv(_: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vdmpyhb.dv.acc.128B"]
    fn vdmpyhb_dv_acc(_: HvxVectorPair, _: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vdmpyhisat.128B"]
    fn vdmpyhisat(_: HvxVectorPair, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdmpyhisat.acc.128B"]
    fn vdmpyhisat_acc(_: HvxVector, _: HvxVectorPair, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdmpyhsat.128B"]
    fn vdmpyhsat(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdmpyhsat.acc.128B"]
    fn vdmpyhsat_acc(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdmpyhsuisat.128B"]
    fn vdmpyhsuisat(_: HvxVectorPair, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdmpyhsuisat.acc.128B"]
    fn vdmpyhsuisat_acc(_: HvxVector, _: HvxVectorPair, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdmpyhsusat.128B"]
    fn vdmpyhsusat(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdmpyhsusat.acc.128B"]
    fn vdmpyhsusat_acc(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdmpyhvsat.128B"]
    fn vdmpyhvsat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdmpyhvsat.acc.128B"]
    fn vdmpyhvsat_acc(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdsaduh.128B"]
    fn vdsaduh(_: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vdsaduh.acc.128B"]
    fn vdsaduh_acc(_: HvxVectorPair, _: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.veqb.128B"]
    fn veqb(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.veqb.and.128B"]
    fn veqb_and(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.veqb.or.128B"]
    fn veqb_or(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.veqb.xor.128B"]
    fn veqb_xor(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.veqh.128B"]
    fn veqh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.veqh.and.128B"]
    fn veqh_and(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.veqh.or.128B"]
    fn veqh_or(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.veqh.xor.128B"]
    fn veqh_xor(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.veqw.128B"]
    fn veqw(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.veqw.and.128B"]
    fn veqw_and(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.veqw.or.128B"]
    fn veqw_or(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.veqw.xor.128B"]
    fn veqw_xor(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vfmax.f8.128B"]
    fn vfmax_f8(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vfmax.hf.128B"]
    fn vfmax_hf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vfmax.sf.128B"]
    fn vfmax_sf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vfmin.f8.128B"]
    fn vfmin_f8(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vfmin.hf.128B"]
    fn vfmin_hf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vfmin.sf.128B"]
    fn vfmin_sf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vfneg.f8.128B"]
    fn vfneg_f8(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vfneg.hf.128B"]
    fn vfneg_hf(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vfneg.sf.128B"]
    fn vfneg_sf(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgathermh.128B"]
    fn vgathermh(_: *mut HvxVector, _: i32, _: i32, _: HvxVector) -> ();
    #[link_name = "llvm.hexagon.V6.vgathermhq.128B"]
    fn vgathermhq(_: *mut HvxVector, _: HvxVector, _: i32, _: i32, _: HvxVector) -> ();
    #[link_name = "llvm.hexagon.V6.vgathermhw.128B"]
    fn vgathermhw(_: *mut HvxVector, _: i32, _: i32, _: HvxVectorPair) -> ();
    #[link_name = "llvm.hexagon.V6.vgathermhwq.128B"]
    fn vgathermhwq(_: *mut HvxVector, _: HvxVector, _: i32, _: i32, _: HvxVectorPair) -> ();
    #[link_name = "llvm.hexagon.V6.vgathermw.128B"]
    fn vgathermw(_: *mut HvxVector, _: i32, _: i32, _: HvxVector) -> ();
    #[link_name = "llvm.hexagon.V6.vgathermwq.128B"]
    fn vgathermwq(_: *mut HvxVector, _: HvxVector, _: i32, _: i32, _: HvxVector) -> ();
    #[link_name = "llvm.hexagon.V6.vgtb.128B"]
    fn vgtb(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgtb.and.128B"]
    fn vgtb_and(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgtb.or.128B"]
    fn vgtb_or(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgtb.xor.128B"]
    fn vgtb_xor(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgth.128B"]
    fn vgth(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgth.and.128B"]
    fn vgth_and(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgth.or.128B"]
    fn vgth_or(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgth.xor.128B"]
    fn vgth_xor(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgthf.128B"]
    fn vgthf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgthf.and.128B"]
    fn vgthf_and(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgthf.or.128B"]
    fn vgthf_or(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgthf.xor.128B"]
    fn vgthf_xor(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgtsf.128B"]
    fn vgtsf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgtsf.and.128B"]
    fn vgtsf_and(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgtsf.or.128B"]
    fn vgtsf_or(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgtsf.xor.128B"]
    fn vgtsf_xor(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgtub.128B"]
    fn vgtub(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgtub.and.128B"]
    fn vgtub_and(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgtub.or.128B"]
    fn vgtub_or(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgtub.xor.128B"]
    fn vgtub_xor(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgtuh.128B"]
    fn vgtuh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgtuh.and.128B"]
    fn vgtuh_and(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgtuh.or.128B"]
    fn vgtuh_or(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgtuh.xor.128B"]
    fn vgtuh_xor(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgtuw.128B"]
    fn vgtuw(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgtuw.and.128B"]
    fn vgtuw_and(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgtuw.or.128B"]
    fn vgtuw_or(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgtuw.xor.128B"]
    fn vgtuw_xor(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgtw.128B"]
    fn vgtw(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgtw.and.128B"]
    fn vgtw_and(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgtw.or.128B"]
    fn vgtw_or(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgtw.xor.128B"]
    fn vgtw_xor(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vinsertwr.128B"]
    fn vinsertwr(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vlalignb.128B"]
    fn vlalignb(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vlalignbi.128B"]
    fn vlalignbi(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vlsrb.128B"]
    fn vlsrb(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vlsrh.128B"]
    fn vlsrh(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vlsrhv.128B"]
    fn vlsrhv(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vlsrw.128B"]
    fn vlsrw(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vlsrwv.128B"]
    fn vlsrwv(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vlutvvb.128B"]
    fn vlutvvb(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vlutvvb.nm.128B"]
    fn vlutvvb_nm(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vlutvvb.oracc.128B"]
    fn vlutvvb_oracc(_: HvxVector, _: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vlutvvb.oracci.128B"]
    fn vlutvvb_oracci(_: HvxVector, _: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vlutvvbi.128B"]
    fn vlutvvbi(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vlutvwh.128B"]
    fn vlutvwh(_: HvxVector, _: HvxVector, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vlutvwh.nm.128B"]
    fn vlutvwh_nm(_: HvxVector, _: HvxVector, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vlutvwh.oracc.128B"]
    fn vlutvwh_oracc(_: HvxVectorPair, _: HvxVector, _: HvxVector, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vlutvwh.oracci.128B"]
    fn vlutvwh_oracci(_: HvxVectorPair, _: HvxVector, _: HvxVector, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vlutvwhi.128B"]
    fn vlutvwhi(_: HvxVector, _: HvxVector, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmax.hf.128B"]
    fn vmax_hf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmax.sf.128B"]
    fn vmax_sf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmaxb.128B"]
    fn vmaxb(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmaxh.128B"]
    fn vmaxh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmaxub.128B"]
    fn vmaxub(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmaxuh.128B"]
    fn vmaxuh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmaxw.128B"]
    fn vmaxw(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmin.hf.128B"]
    fn vmin_hf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmin.sf.128B"]
    fn vmin_sf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vminb.128B"]
    fn vminb(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vminh.128B"]
    fn vminh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vminub.128B"]
    fn vminub(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vminuh.128B"]
    fn vminuh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vminw.128B"]
    fn vminw(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpabus.128B"]
    fn vmpabus(_: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpabus.acc.128B"]
    fn vmpabus_acc(_: HvxVectorPair, _: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpabusv.128B"]
    fn vmpabusv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpabuu.128B"]
    fn vmpabuu(_: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpabuu.acc.128B"]
    fn vmpabuu_acc(_: HvxVectorPair, _: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpabuuv.128B"]
    fn vmpabuuv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpahb.128B"]
    fn vmpahb(_: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpahb.acc.128B"]
    fn vmpahb_acc(_: HvxVectorPair, _: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpauhb.128B"]
    fn vmpauhb(_: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpauhb.acc.128B"]
    fn vmpauhb_acc(_: HvxVectorPair, _: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpy.hf.hf.128B"]
    fn vmpy_hf_hf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpy.hf.hf.acc.128B"]
    fn vmpy_hf_hf_acc(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpy.qf16.128B"]
    fn vmpy_qf16(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpy.qf16.hf.128B"]
    fn vmpy_qf16_hf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpy.qf16.mix.hf.128B"]
    fn vmpy_qf16_mix_hf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpy.qf32.128B"]
    fn vmpy_qf32(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpy.qf32.hf.128B"]
    fn vmpy_qf32_hf(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpy.qf32.mix.hf.128B"]
    fn vmpy_qf32_mix_hf(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpy.qf32.qf16.128B"]
    fn vmpy_qf32_qf16(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpy.qf32.sf.128B"]
    fn vmpy_qf32_sf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpy.sf.hf.128B"]
    fn vmpy_sf_hf(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpy.sf.hf.acc.128B"]
    fn vmpy_sf_hf_acc(_: HvxVectorPair, _: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpy.sf.sf.128B"]
    fn vmpy_sf_sf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpybus.128B"]
    fn vmpybus(_: HvxVector, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpybus.acc.128B"]
    fn vmpybus_acc(_: HvxVectorPair, _: HvxVector, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpybusv.128B"]
    fn vmpybusv(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpybusv.acc.128B"]
    fn vmpybusv_acc(_: HvxVectorPair, _: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpybv.128B"]
    fn vmpybv(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpybv.acc.128B"]
    fn vmpybv_acc(_: HvxVectorPair, _: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyewuh.128B"]
    fn vmpyewuh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyewuh.64.128B"]
    fn vmpyewuh_64(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyh.128B"]
    fn vmpyh(_: HvxVector, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyh.acc.128B"]
    fn vmpyh_acc(_: HvxVectorPair, _: HvxVector, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyhsat.acc.128B"]
    fn vmpyhsat_acc(_: HvxVectorPair, _: HvxVector, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyhsrs.128B"]
    fn vmpyhsrs(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyhss.128B"]
    fn vmpyhss(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyhus.128B"]
    fn vmpyhus(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyhus.acc.128B"]
    fn vmpyhus_acc(_: HvxVectorPair, _: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyhv.128B"]
    fn vmpyhv(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyhv.acc.128B"]
    fn vmpyhv_acc(_: HvxVectorPair, _: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyhvsrs.128B"]
    fn vmpyhvsrs(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyieoh.128B"]
    fn vmpyieoh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyiewh.acc.128B"]
    fn vmpyiewh_acc(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyiewuh.128B"]
    fn vmpyiewuh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyiewuh.acc.128B"]
    fn vmpyiewuh_acc(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyih.128B"]
    fn vmpyih(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyih.acc.128B"]
    fn vmpyih_acc(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyihb.128B"]
    fn vmpyihb(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyihb.acc.128B"]
    fn vmpyihb_acc(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyiowh.128B"]
    fn vmpyiowh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyiwb.128B"]
    fn vmpyiwb(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyiwb.acc.128B"]
    fn vmpyiwb_acc(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyiwh.128B"]
    fn vmpyiwh(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyiwh.acc.128B"]
    fn vmpyiwh_acc(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyiwub.128B"]
    fn vmpyiwub(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyiwub.acc.128B"]
    fn vmpyiwub_acc(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyowh.128B"]
    fn vmpyowh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyowh.64.acc.128B"]
    fn vmpyowh_64_acc(_: HvxVectorPair, _: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyowh.rnd.128B"]
    fn vmpyowh_rnd(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyowh.rnd.sacc.128B"]
    fn vmpyowh_rnd_sacc(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyowh.sacc.128B"]
    fn vmpyowh_sacc(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyub.128B"]
    fn vmpyub(_: HvxVector, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyub.acc.128B"]
    fn vmpyub_acc(_: HvxVectorPair, _: HvxVector, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyubv.128B"]
    fn vmpyubv(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyubv.acc.128B"]
    fn vmpyubv_acc(_: HvxVectorPair, _: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyuh.128B"]
    fn vmpyuh(_: HvxVector, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyuh.acc.128B"]
    fn vmpyuh_acc(_: HvxVectorPair, _: HvxVector, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyuhe.128B"]
    fn vmpyuhe(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyuhe.acc.128B"]
    fn vmpyuhe_acc(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyuhv.128B"]
    fn vmpyuhv(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyuhv.acc.128B"]
    fn vmpyuhv_acc(_: HvxVectorPair, _: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyuhvs.128B"]
    fn vmpyuhvs(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmux.128B"]
    fn vmux(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vnavgb.128B"]
    fn vnavgb(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vnavgh.128B"]
    fn vnavgh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vnavgub.128B"]
    fn vnavgub(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vnavgw.128B"]
    fn vnavgw(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vnormamth.128B"]
    fn vnormamth(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vnormamtw.128B"]
    fn vnormamtw(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vnot.128B"]
    fn vnot(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vor.128B"]
    fn vor(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vpackeb.128B"]
    fn vpackeb(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vpackeh.128B"]
    fn vpackeh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vpackhb.sat.128B"]
    fn vpackhb_sat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vpackhub.sat.128B"]
    fn vpackhub_sat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vpackob.128B"]
    fn vpackob(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vpackoh.128B"]
    fn vpackoh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vpackwh.sat.128B"]
    fn vpackwh_sat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vpackwuh.sat.128B"]
    fn vpackwuh_sat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vpopcounth.128B"]
    fn vpopcounth(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vprefixqb.128B"]
    fn vprefixqb(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vprefixqh.128B"]
    fn vprefixqh(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vprefixqw.128B"]
    fn vprefixqw(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vrdelta.128B"]
    fn vrdelta(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vrmpybus.128B"]
    fn vrmpybus(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vrmpybus.acc.128B"]
    fn vrmpybus_acc(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vrmpybusi.128B"]
    fn vrmpybusi(_: HvxVectorPair, _: i32, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vrmpybusi.acc.128B"]
    fn vrmpybusi_acc(_: HvxVectorPair, _: HvxVectorPair, _: i32, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vrmpybusv.128B"]
    fn vrmpybusv(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vrmpybusv.acc.128B"]
    fn vrmpybusv_acc(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vrmpybv.128B"]
    fn vrmpybv(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vrmpybv.acc.128B"]
    fn vrmpybv_acc(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vrmpyub.128B"]
    fn vrmpyub(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vrmpyub.acc.128B"]
    fn vrmpyub_acc(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vrmpyubi.128B"]
    fn vrmpyubi(_: HvxVectorPair, _: i32, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vrmpyubi.acc.128B"]
    fn vrmpyubi_acc(_: HvxVectorPair, _: HvxVectorPair, _: i32, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vrmpyubv.128B"]
    fn vrmpyubv(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vrmpyubv.acc.128B"]
    fn vrmpyubv_acc(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vror.128B"]
    fn vror(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vrotr.128B"]
    fn vrotr(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vroundhb.128B"]
    fn vroundhb(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vroundhub.128B"]
    fn vroundhub(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vrounduhub.128B"]
    fn vrounduhub(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vrounduwuh.128B"]
    fn vrounduwuh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vroundwh.128B"]
    fn vroundwh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vroundwuh.128B"]
    fn vroundwuh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vrsadubi.128B"]
    fn vrsadubi(_: HvxVectorPair, _: i32, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vrsadubi.acc.128B"]
    fn vrsadubi_acc(_: HvxVectorPair, _: HvxVectorPair, _: i32, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vsatdw.128B"]
    fn vsatdw(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsathub.128B"]
    fn vsathub(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsatuwuh.128B"]
    fn vsatuwuh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsatwh.128B"]
    fn vsatwh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsb.128B"]
    fn vsb(_: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vscattermh.128B"]
    fn vscattermh(_: i32, _: i32, _: HvxVector, _: HvxVector) -> ();
    #[link_name = "llvm.hexagon.V6.vscattermh.add.128B"]
    fn vscattermh_add(_: i32, _: i32, _: HvxVector, _: HvxVector) -> ();
    #[link_name = "llvm.hexagon.V6.vscattermhq.128B"]
    fn vscattermhq(_: HvxVector, _: i32, _: i32, _: HvxVector, _: HvxVector) -> ();
    #[link_name = "llvm.hexagon.V6.vscattermhw.128B"]
    fn vscattermhw(_: i32, _: i32, _: HvxVectorPair, _: HvxVector) -> ();
    #[link_name = "llvm.hexagon.V6.vscattermhw.add.128B"]
    fn vscattermhw_add(_: i32, _: i32, _: HvxVectorPair, _: HvxVector) -> ();
    #[link_name = "llvm.hexagon.V6.vscattermhwq.128B"]
    fn vscattermhwq(_: HvxVector, _: i32, _: i32, _: HvxVectorPair, _: HvxVector) -> ();
    #[link_name = "llvm.hexagon.V6.vscattermw.128B"]
    fn vscattermw(_: i32, _: i32, _: HvxVector, _: HvxVector) -> ();
    #[link_name = "llvm.hexagon.V6.vscattermw.add.128B"]
    fn vscattermw_add(_: i32, _: i32, _: HvxVector, _: HvxVector) -> ();
    #[link_name = "llvm.hexagon.V6.vscattermwq.128B"]
    fn vscattermwq(_: HvxVector, _: i32, _: i32, _: HvxVector, _: HvxVector) -> ();
    #[link_name = "llvm.hexagon.V6.vsh.128B"]
    fn vsh(_: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vshufeh.128B"]
    fn vshufeh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vshuffb.128B"]
    fn vshuffb(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vshuffeb.128B"]
    fn vshuffeb(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vshuffh.128B"]
    fn vshuffh(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vshuffob.128B"]
    fn vshuffob(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vshuffvdd.128B"]
    fn vshuffvdd(_: HvxVector, _: HvxVector, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vshufoeb.128B"]
    fn vshufoeb(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vshufoeh.128B"]
    fn vshufoeh(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vshufoh.128B"]
    fn vshufoh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsub.hf.128B"]
    fn vsub_hf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsub.hf.hf.128B"]
    fn vsub_hf_hf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsub.qf16.128B"]
    fn vsub_qf16(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsub.qf16.mix.128B"]
    fn vsub_qf16_mix(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsub.qf32.128B"]
    fn vsub_qf32(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsub.qf32.mix.128B"]
    fn vsub_qf32_mix(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsub.sf.128B"]
    fn vsub_sf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsub.sf.hf.128B"]
    fn vsub_sf_hf(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vsub.sf.sf.128B"]
    fn vsub_sf_sf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsubb.128B"]
    fn vsubb(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsubb.dv.128B"]
    fn vsubb_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vsubbnq.128B"]
    fn vsubbnq(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsubbq.128B"]
    fn vsubbq(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsubbsat.128B"]
    fn vsubbsat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsubbsat.dv.128B"]
    fn vsubbsat_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vsubh.128B"]
    fn vsubh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsubh.dv.128B"]
    fn vsubh_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vsubhnq.128B"]
    fn vsubhnq(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsubhq.128B"]
    fn vsubhq(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsubhsat.128B"]
    fn vsubhsat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsubhsat.dv.128B"]
    fn vsubhsat_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vsubhw.128B"]
    fn vsubhw(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vsububh.128B"]
    fn vsububh(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vsububsat.128B"]
    fn vsububsat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsububsat.dv.128B"]
    fn vsububsat_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vsubububb.sat.128B"]
    fn vsubububb_sat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsubuhsat.128B"]
    fn vsubuhsat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsubuhsat.dv.128B"]
    fn vsubuhsat_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vsubuhw.128B"]
    fn vsubuhw(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vsubuwsat.128B"]
    fn vsubuwsat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsubuwsat.dv.128B"]
    fn vsubuwsat_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vsubw.128B"]
    fn vsubw(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsubw.dv.128B"]
    fn vsubw_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vsubwnq.128B"]
    fn vsubwnq(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsubwq.128B"]
    fn vsubwq(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsubwsat.128B"]
    fn vsubwsat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsubwsat.dv.128B"]
    fn vsubwsat_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vswap.128B"]
    fn vswap(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vtmpyb.128B"]
    fn vtmpyb(_: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vtmpyb.acc.128B"]
    fn vtmpyb_acc(_: HvxVectorPair, _: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vtmpybus.128B"]
    fn vtmpybus(_: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vtmpybus.acc.128B"]
    fn vtmpybus_acc(_: HvxVectorPair, _: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vtmpyhb.128B"]
    fn vtmpyhb(_: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vtmpyhb.acc.128B"]
    fn vtmpyhb_acc(_: HvxVectorPair, _: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vunpackb.128B"]
    fn vunpackb(_: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vunpackh.128B"]
    fn vunpackh(_: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vunpackob.128B"]
    fn vunpackob(_: HvxVectorPair, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vunpackoh.128B"]
    fn vunpackoh(_: HvxVectorPair, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vunpackub.128B"]
    fn vunpackub(_: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vunpackuh.128B"]
    fn vunpackuh(_: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vxor.128B"]
    fn vxor(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vzb.128B"]
    fn vzb(_: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vzh.128B"]
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
pub unsafe fn q6_r_vextract_vr(vu: HvxVector, rs: i32) -> i32 {
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
pub unsafe fn q6_v_hi_w(vss: HvxVectorPair) -> HvxVector {
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
pub unsafe fn q6_v_lo_w(vss: HvxVectorPair) -> HvxVector {
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
pub unsafe fn q6_v_vsplat_r(rt: i32) -> HvxVector {
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
pub unsafe fn q6_vuh_vabsdiff_vhvh(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vub_vabsdiff_vubvub(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vuh_vabsdiff_vuhvuh(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vuw_vabsdiff_vwvw(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vh_vabs_vh(vu: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vh_vabs_vh_sat(vu: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vw_vabs_vw(vu: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vw_vabs_vw_sat(vu: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vb_vadd_vbvb(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_wb_vadd_wbwb(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
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
pub unsafe fn q6_vh_vadd_vhvh(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_wh_vadd_whwh(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
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
pub unsafe fn q6_vh_vadd_vhvh_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_wh_vadd_whwh_sat(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
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
pub unsafe fn q6_ww_vadd_vhvh(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
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
pub unsafe fn q6_wh_vadd_vubvub(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
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
pub unsafe fn q6_vub_vadd_vubvub_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_wub_vadd_wubwub_sat(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
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
pub unsafe fn q6_vuh_vadd_vuhvuh_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_wuh_vadd_wuhwuh_sat(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
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
pub unsafe fn q6_ww_vadd_vuhvuh(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
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
pub unsafe fn q6_vw_vadd_vwvw(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_ww_vadd_wwww(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
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
pub unsafe fn q6_vw_vadd_vwvw_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_ww_vadd_wwww_sat(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
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
pub unsafe fn q6_v_valign_vvr(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_v_valign_vvi(vu: HvxVector, vv: HvxVector, iu3: i32) -> HvxVector {
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
pub unsafe fn q6_v_vand_vv(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vh_vasl_vhr(vu: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_vh_vasl_vhvh(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vw_vasl_vwr(vu: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_vw_vaslacc_vwvwr(vx: HvxVector, vu: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_vw_vasl_vwvw(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vh_vasr_vhr(vu: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_vb_vasr_vhvhr_rnd_sat(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_vub_vasr_vhvhr_rnd_sat(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_vub_vasr_vhvhr_sat(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_vh_vasr_vhvh(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vw_vasr_vwr(vu: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_vw_vasracc_vwvwr(vx: HvxVector, vu: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_vh_vasr_vwvwr(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_vh_vasr_vwvwr_rnd_sat(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_vh_vasr_vwvwr_sat(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_vuh_vasr_vwvwr_sat(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_vw_vasr_vwvw(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_v_equals_v(vu: HvxVector) -> HvxVector {
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
pub unsafe fn q6_w_equals_w(vuu: HvxVectorPair) -> HvxVectorPair {
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
pub unsafe fn q6_vh_vavg_vhvh(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vh_vavg_vhvh_rnd(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vub_vavg_vubvub(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vub_vavg_vubvub_rnd(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vuh_vavg_vuhvuh(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vuh_vavg_vuhvuh_rnd(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vw_vavg_vwvw(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vw_vavg_vwvw_rnd(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vuh_vcl0_vuh(vu: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vuw_vcl0_vuw(vu: HvxVector) -> HvxVector {
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
pub unsafe fn q6_w_vcombine_vv(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
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
pub unsafe fn q6_v_vzero() -> HvxVector {
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
pub unsafe fn q6_vb_vdeal_vb(vu: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vb_vdeale_vbvb(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vh_vdeal_vh(vu: HvxVector) -> HvxVector {
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
pub unsafe fn q6_w_vdeal_vvr(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVectorPair {
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
pub unsafe fn q6_v_vdelta_vv(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vh_vdmpy_vubrb(vu: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_vh_vdmpyacc_vhvubrb(vx: HvxVector, vu: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_wh_vdmpy_wubrb(vuu: HvxVectorPair, rt: i32) -> HvxVectorPair {
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
pub unsafe fn q6_wh_vdmpyacc_whwubrb(
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
pub unsafe fn q6_vw_vdmpy_vhrb(vu: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_vw_vdmpyacc_vwvhrb(vx: HvxVector, vu: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_ww_vdmpy_whrb(vuu: HvxVectorPair, rt: i32) -> HvxVectorPair {
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
pub unsafe fn q6_ww_vdmpyacc_wwwhrb(
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
pub unsafe fn q6_vw_vdmpy_whrh_sat(vuu: HvxVectorPair, rt: i32) -> HvxVector {
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
pub unsafe fn q6_vw_vdmpyacc_vwwhrh_sat(vx: HvxVector, vuu: HvxVectorPair, rt: i32) -> HvxVector {
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
pub unsafe fn q6_vw_vdmpy_vhrh_sat(vu: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_vw_vdmpyacc_vwvhrh_sat(vx: HvxVector, vu: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_vw_vdmpy_whruh_sat(vuu: HvxVectorPair, rt: i32) -> HvxVector {
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
pub unsafe fn q6_vw_vdmpyacc_vwwhruh_sat(vx: HvxVector, vuu: HvxVectorPair, rt: i32) -> HvxVector {
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
pub unsafe fn q6_vw_vdmpy_vhruh_sat(vu: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_vw_vdmpyacc_vwvhruh_sat(vx: HvxVector, vu: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_vw_vdmpy_vhvh_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vw_vdmpyacc_vwvhvh_sat(vx: HvxVector, vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_wuw_vdsad_wuhruh(vuu: HvxVectorPair, rt: i32) -> HvxVectorPair {
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
pub unsafe fn q6_wuw_vdsadacc_wuwwuhruh(
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
pub unsafe fn q6_vw_vinsert_vwr(vx: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_v_vlalign_vvr(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_v_vlalign_vvi(vu: HvxVector, vv: HvxVector, iu3: i32) -> HvxVector {
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
pub unsafe fn q6_vuh_vlsr_vuhr(vu: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_vh_vlsr_vhvh(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vuw_vlsr_vuwr(vu: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_vw_vlsr_vwvw(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vb_vlut32_vbvbr(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_vb_vlut32or_vbvbvbr(
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
pub unsafe fn q6_wh_vlut16_vbvhr(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVectorPair {
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
pub unsafe fn q6_wh_vlut16or_whvbvhr(
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
pub unsafe fn q6_vh_vmax_vhvh(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vub_vmax_vubvub(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vuh_vmax_vuhvuh(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vw_vmax_vwvw(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vh_vmin_vhvh(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vub_vmin_vubvub(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vuh_vmin_vuhvuh(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vw_vmin_vwvw(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_wh_vmpa_wubrb(vuu: HvxVectorPair, rt: i32) -> HvxVectorPair {
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
pub unsafe fn q6_wh_vmpaacc_whwubrb(
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
pub unsafe fn q6_wh_vmpa_wubwb(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
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
pub unsafe fn q6_wh_vmpa_wubwub(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
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
pub unsafe fn q6_ww_vmpa_whrb(vuu: HvxVectorPair, rt: i32) -> HvxVectorPair {
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
pub unsafe fn q6_ww_vmpaacc_wwwhrb(
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
pub unsafe fn q6_wh_vmpy_vubrb(vu: HvxVector, rt: i32) -> HvxVectorPair {
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
pub unsafe fn q6_wh_vmpyacc_whvubrb(vxx: HvxVectorPair, vu: HvxVector, rt: i32) -> HvxVectorPair {
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
pub unsafe fn q6_wh_vmpy_vubvb(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
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
pub unsafe fn q6_wh_vmpyacc_whvubvb(
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
pub unsafe fn q6_wh_vmpy_vbvb(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
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
pub unsafe fn q6_wh_vmpyacc_whvbvb(
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
pub unsafe fn q6_vw_vmpye_vwvuh(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_ww_vmpy_vhrh(vu: HvxVector, rt: i32) -> HvxVectorPair {
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
pub unsafe fn q6_ww_vmpyacc_wwvhrh_sat(
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
pub unsafe fn q6_vh_vmpy_vhrh_s1_rnd_sat(vu: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_vh_vmpy_vhrh_s1_sat(vu: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_ww_vmpy_vhvuh(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
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
pub unsafe fn q6_ww_vmpyacc_wwvhvuh(
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
pub unsafe fn q6_ww_vmpy_vhvh(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
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
pub unsafe fn q6_ww_vmpyacc_wwvhvh(
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
pub unsafe fn q6_vh_vmpy_vhvh_s1_rnd_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vw_vmpyieo_vhvh(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vw_vmpyieacc_vwvwvh(vx: HvxVector, vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vw_vmpyie_vwvuh(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vw_vmpyieacc_vwvwvuh(vx: HvxVector, vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vh_vmpyi_vhvh(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vh_vmpyiacc_vhvhvh(vx: HvxVector, vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vh_vmpyi_vhrb(vu: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_vh_vmpyiacc_vhvhrb(vx: HvxVector, vu: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_vw_vmpyio_vwvh(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vw_vmpyi_vwrb(vu: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_vw_vmpyiacc_vwvwrb(vx: HvxVector, vu: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_vw_vmpyi_vwrh(vu: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_vw_vmpyiacc_vwvwrh(vx: HvxVector, vu: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_vw_vmpyo_vwvh_s1_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vw_vmpyo_vwvh_s1_rnd_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vw_vmpyoacc_vwvwvh_s1_rnd_sat_shift(
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
pub unsafe fn q6_vw_vmpyoacc_vwvwvh_s1_sat_shift(
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
pub unsafe fn q6_wuh_vmpy_vubrub(vu: HvxVector, rt: i32) -> HvxVectorPair {
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
pub unsafe fn q6_wuh_vmpyacc_wuhvubrub(
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
pub unsafe fn q6_wuh_vmpy_vubvub(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
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
pub unsafe fn q6_wuh_vmpyacc_wuhvubvub(
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
pub unsafe fn q6_wuw_vmpy_vuhruh(vu: HvxVector, rt: i32) -> HvxVectorPair {
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
pub unsafe fn q6_wuw_vmpyacc_wuwvuhruh(
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
pub unsafe fn q6_wuw_vmpy_vuhvuh(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
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
pub unsafe fn q6_wuw_vmpyacc_wuwvuhvuh(
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
pub unsafe fn q6_vh_vnavg_vhvh(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vb_vnavg_vubvub(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vw_vnavg_vwvw(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vh_vnormamt_vh(vu: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vw_vnormamt_vw(vu: HvxVector) -> HvxVector {
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
pub unsafe fn q6_v_vnot_v(vu: HvxVector) -> HvxVector {
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
pub unsafe fn q6_v_vor_vv(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vb_vpacke_vhvh(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vh_vpacke_vwvw(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vb_vpack_vhvh_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vub_vpack_vhvh_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vb_vpacko_vhvh(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vh_vpacko_vwvw(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vh_vpack_vwvw_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vuh_vpack_vwvw_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vh_vpopcount_vh(vu: HvxVector) -> HvxVector {
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
pub unsafe fn q6_v_vrdelta_vv(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vw_vrmpy_vubrb(vu: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_vw_vrmpyacc_vwvubrb(vx: HvxVector, vu: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_ww_vrmpy_wubrbi(vuu: HvxVectorPair, rt: i32, iu1: i32) -> HvxVectorPair {
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
pub unsafe fn q6_ww_vrmpyacc_wwwubrbi(
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
pub unsafe fn q6_vw_vrmpy_vubvb(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vw_vrmpyacc_vwvubvb(vx: HvxVector, vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vw_vrmpy_vbvb(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vw_vrmpyacc_vwvbvb(vx: HvxVector, vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vuw_vrmpy_vubrub(vu: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_vuw_vrmpyacc_vuwvubrub(vx: HvxVector, vu: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_wuw_vrmpy_wubrubi(vuu: HvxVectorPair, rt: i32, iu1: i32) -> HvxVectorPair {
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
pub unsafe fn q6_wuw_vrmpyacc_wuwwubrubi(
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
pub unsafe fn q6_vuw_vrmpy_vubvub(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vuw_vrmpyacc_vuwvubvub(vx: HvxVector, vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_v_vror_vr(vu: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_vb_vround_vhvh_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vub_vround_vhvh_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vh_vround_vwvw_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vuh_vround_vwvw_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_wuw_vrsad_wubrubi(vuu: HvxVectorPair, rt: i32, iu1: i32) -> HvxVectorPair {
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
pub unsafe fn q6_wuw_vrsadacc_wuwwubrubi(
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
pub unsafe fn q6_vub_vsat_vhvh(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vh_vsat_vwvw(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_wh_vsxt_vb(vu: HvxVector) -> HvxVectorPair {
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
pub unsafe fn q6_ww_vsxt_vh(vu: HvxVector) -> HvxVectorPair {
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
pub unsafe fn q6_vh_vshuffe_vhvh(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vb_vshuff_vb(vu: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vb_vshuffe_vbvb(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vh_vshuff_vh(vu: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vb_vshuffo_vbvb(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_w_vshuff_vvr(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVectorPair {
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
pub unsafe fn q6_wb_vshuffoe_vbvb(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
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
pub unsafe fn q6_wh_vshuffoe_vhvh(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
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
pub unsafe fn q6_vh_vshuffo_vhvh(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vb_vsub_vbvb(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_wb_vsub_wbwb(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
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
pub unsafe fn q6_vh_vsub_vhvh(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_wh_vsub_whwh(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
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
pub unsafe fn q6_vh_vsub_vhvh_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_wh_vsub_whwh_sat(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
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
pub unsafe fn q6_ww_vsub_vhvh(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
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
pub unsafe fn q6_wh_vsub_vubvub(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
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
pub unsafe fn q6_vub_vsub_vubvub_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_wub_vsub_wubwub_sat(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
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
pub unsafe fn q6_vuh_vsub_vuhvuh_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_wuh_vsub_wuhwuh_sat(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
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
pub unsafe fn q6_ww_vsub_vuhvuh(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
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
pub unsafe fn q6_vw_vsub_vwvw(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_ww_vsub_wwww(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
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
pub unsafe fn q6_vw_vsub_vwvw_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_ww_vsub_wwww_sat(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
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
pub unsafe fn q6_wh_vtmpy_wbrb(vuu: HvxVectorPair, rt: i32) -> HvxVectorPair {
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
pub unsafe fn q6_wh_vtmpyacc_whwbrb(
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
pub unsafe fn q6_wh_vtmpy_wubrb(vuu: HvxVectorPair, rt: i32) -> HvxVectorPair {
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
pub unsafe fn q6_wh_vtmpyacc_whwubrb(
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
pub unsafe fn q6_ww_vtmpy_whrb(vuu: HvxVectorPair, rt: i32) -> HvxVectorPair {
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
pub unsafe fn q6_ww_vtmpyacc_wwwhrb(
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
pub unsafe fn q6_wh_vunpack_vb(vu: HvxVector) -> HvxVectorPair {
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
pub unsafe fn q6_ww_vunpack_vh(vu: HvxVector) -> HvxVectorPair {
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
pub unsafe fn q6_wh_vunpackoor_whvb(vxx: HvxVectorPair, vu: HvxVector) -> HvxVectorPair {
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
pub unsafe fn q6_ww_vunpackoor_wwvh(vxx: HvxVectorPair, vu: HvxVector) -> HvxVectorPair {
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
pub unsafe fn q6_wuh_vunpack_vub(vu: HvxVector) -> HvxVectorPair {
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
pub unsafe fn q6_wuw_vunpack_vuh(vu: HvxVector) -> HvxVectorPair {
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
pub unsafe fn q6_v_vxor_vv(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_wuh_vzxt_vub(vu: HvxVector) -> HvxVectorPair {
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
pub unsafe fn q6_wuw_vzxt_vuh(vu: HvxVector) -> HvxVectorPair {
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
pub unsafe fn q6_vb_vsplat_r(rt: i32) -> HvxVector {
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
pub unsafe fn q6_vh_vsplat_r(rt: i32) -> HvxVector {
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
pub unsafe fn q6_vb_vadd_vbvb_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_wb_vadd_wbwb_sat(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
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
pub unsafe fn q6_vh_vadd_vclb_vhvh(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vw_vadd_vclb_vwvw(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_ww_vaddacc_wwvhvh(
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
pub unsafe fn q6_wh_vaddacc_whvubvub(
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
pub unsafe fn q6_vub_vadd_vubvb_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_ww_vaddacc_wwvuhvuh(
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
pub unsafe fn q6_vuw_vadd_vuwvuw_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_wuw_vadd_wuwwuw_sat(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
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
pub unsafe fn q6_vb_vasr_vhvhr_sat(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_vuh_vasr_vuwvuwr_rnd_sat(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_vuh_vasr_vwvwr_rnd_sat(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_vub_vlsr_vubr(vu: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_vb_vlut32_vbvbr_nomatch(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_vb_vlut32or_vbvbvbi(
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
pub unsafe fn q6_vb_vlut32_vbvbi(vu: HvxVector, vv: HvxVector, iu3: i32) -> HvxVector {
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
pub unsafe fn q6_wh_vlut16_vbvhr_nomatch(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVectorPair {
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
pub unsafe fn q6_wh_vlut16or_whvbvhi(
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
pub unsafe fn q6_wh_vlut16_vbvhi(vu: HvxVector, vv: HvxVector, iu3: i32) -> HvxVectorPair {
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
pub unsafe fn q6_vb_vmax_vbvb(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vb_vmin_vbvb(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_ww_vmpa_wuhrb(vuu: HvxVectorPair, rt: i32) -> HvxVectorPair {
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
pub unsafe fn q6_ww_vmpaacc_wwwuhrb(
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
pub unsafe fn q6_w_vmpye_vwvuh(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
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
pub unsafe fn q6_vw_vmpyi_vwrub(vu: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_vw_vmpyiacc_vwvwrub(vx: HvxVector, vu: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_w_vmpyoacc_wvwvh(
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
pub unsafe fn q6_vub_vround_vuhvuh_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vuh_vround_vuwvuw_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vuh_vsat_vuwvuw(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vb_vsub_vbvb_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_wb_vsub_wbwb_sat(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
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
pub unsafe fn q6_vub_vsub_vubvb_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vuw_vsub_vuwvuw_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_wuw_vsub_wuwwuw_sat(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
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
pub unsafe fn q6_vb_vabs_vb(vu: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vb_vabs_vb_sat(vu: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vh_vaslacc_vhvhr(vx: HvxVector, vu: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_vh_vasracc_vhvhr(vx: HvxVector, vu: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_vub_vasr_vuhvuhr_rnd_sat(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_vub_vasr_vuhvuhr_sat(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_vuh_vasr_vuwvuwr_sat(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_vb_vavg_vbvb(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vb_vavg_vbvb_rnd(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vuw_vavg_vuwvuw(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vuw_vavg_vuwvuw_rnd(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_w_vzero() -> HvxVectorPair {
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
pub unsafe fn q6_vgather_armvh(rs: *mut HvxVector, rt: i32, mu: i32, vv: HvxVector) {
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
pub unsafe fn q6_vgather_armww(rs: *mut HvxVector, rt: i32, mu: i32, vvv: HvxVectorPair) {
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
pub unsafe fn q6_vgather_armvw(rs: *mut HvxVector, rt: i32, mu: i32, vv: HvxVector) {
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
pub unsafe fn q6_wh_vmpa_wubrub(vuu: HvxVectorPair, rt: i32) -> HvxVectorPair {
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
pub unsafe fn q6_wh_vmpaacc_whwubrub(
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
pub unsafe fn q6_ww_vmpyacc_wwvhrh(vxx: HvxVectorPair, vu: HvxVector, rt: i32) -> HvxVectorPair {
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
pub unsafe fn q6_vuw_vmpye_vuhruh(vu: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_vuw_vmpyeacc_vuwvuhruh(vx: HvxVector, vu: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_vb_vnavg_vbvb(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vscatter_rmvhv(rt: i32, mu: i32, vv: HvxVector, vw: HvxVector) {
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
pub unsafe fn q6_vscatteracc_rmvhv(rt: i32, mu: i32, vv: HvxVector, vw: HvxVector) {
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
pub unsafe fn q6_vscatter_rmwwv(rt: i32, mu: i32, vvv: HvxVectorPair, vw: HvxVector) {
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
pub unsafe fn q6_vscatteracc_rmwwv(rt: i32, mu: i32, vvv: HvxVectorPair, vw: HvxVector) {
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
pub unsafe fn q6_vscatter_rmvwv(rt: i32, mu: i32, vv: HvxVector, vw: HvxVector) {
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
pub unsafe fn q6_vscatteracc_rmvwv(rt: i32, mu: i32, vv: HvxVector, vw: HvxVector) {
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
pub unsafe fn q6_ww_vasrinto_wwvwvw(
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
pub unsafe fn q6_vuw_vrotr_vuwvuw(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vw_vsatdw_vwvw(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_ww_v6mpy_wubwbi_h(
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
pub unsafe fn q6_ww_v6mpyacc_wwwubwbi_h(
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
pub unsafe fn q6_ww_v6mpy_wubwbi_v(
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
pub unsafe fn q6_ww_v6mpyacc_wwwubwbi_v(
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
pub unsafe fn q6_vhf_vabs_vhf(vu: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vsf_vabs_vsf(vu: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vqf16_vadd_vhfvhf(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vhf_vadd_vhfvhf(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vqf16_vadd_vqf16vqf16(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vqf16_vadd_vqf16vhf(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vqf32_vadd_vqf32vqf32(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vqf32_vadd_vqf32vsf(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vqf32_vadd_vsfvsf(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_wsf_vadd_vhfvhf(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
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
pub unsafe fn q6_vsf_vadd_vsfvsf(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vw_vfmv_vw(vu: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vhf_equals_vqf16(vu: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vhf_equals_wqf32(vuu: HvxVectorPair) -> HvxVector {
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
pub unsafe fn q6_vsf_equals_vqf32(vu: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vb_vcvt_vhfvhf(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vh_vcvt_vhf(vu: HvxVector) -> HvxVector {
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
pub unsafe fn q6_whf_vcvt_vb(vu: HvxVector) -> HvxVectorPair {
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
pub unsafe fn q6_vhf_vcvt_vh(vu: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vhf_vcvt_vsfvsf(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_whf_vcvt_vub(vu: HvxVector) -> HvxVectorPair {
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
pub unsafe fn q6_vhf_vcvt_vuh(vu: HvxVector) -> HvxVector {
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
pub unsafe fn q6_wsf_vcvt_vhf(vu: HvxVector) -> HvxVectorPair {
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
pub unsafe fn q6_vub_vcvt_vhfvhf(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vuh_vcvt_vhf(vu: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vsf_vdmpy_vhfvhf(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vsf_vdmpyacc_vsfvhfvhf(vx: HvxVector, vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vhf_vfmax_vhfvhf(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vsf_vfmax_vsfvsf(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vhf_vfmin_vhfvhf(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vsf_vfmin_vsfvsf(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vhf_vfneg_vhf(vu: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vsf_vfneg_vsf(vu: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vhf_vmax_vhfvhf(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vsf_vmax_vsfvsf(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vhf_vmin_vhfvhf(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vsf_vmin_vsfvsf(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vhf_vmpy_vhfvhf(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vhf_vmpyacc_vhfvhfvhf(vx: HvxVector, vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vqf16_vmpy_vqf16vqf16(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vqf16_vmpy_vhfvhf(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vqf16_vmpy_vqf16vhf(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vqf32_vmpy_vqf32vqf32(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_wqf32_vmpy_vhfvhf(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
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
pub unsafe fn q6_wqf32_vmpy_vqf16vhf(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
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
pub unsafe fn q6_wqf32_vmpy_vqf16vqf16(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
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
pub unsafe fn q6_vqf32_vmpy_vsfvsf(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_wsf_vmpy_vhfvhf(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
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
pub unsafe fn q6_wsf_vmpyacc_wsfvhfvhf(
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
pub unsafe fn q6_vsf_vmpy_vsfvsf(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vqf16_vsub_vhfvhf(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vhf_vsub_vhfvhf(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vqf16_vsub_vqf16vqf16(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vqf16_vsub_vqf16vhf(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vqf32_vsub_vqf32vqf32(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vqf32_vsub_vqf32vsf(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vqf32_vsub_vsfvsf(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_wsf_vsub_vhfvhf(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
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
pub unsafe fn q6_vsf_vsub_vsfvsf(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vub_vasr_wuhvub_rnd_sat(vuu: HvxVectorPair, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vub_vasr_wuhvub_sat(vuu: HvxVectorPair, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vuh_vasr_wwvuh_rnd_sat(vuu: HvxVectorPair, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vuh_vasr_wwvuh_sat(vuu: HvxVectorPair, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vuh_vmpy_vuhvuh_rs16(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vh_equals_vhf(vu: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vhf_equals_vh(vu: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vsf_equals_vw(vu: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vw_equals_vsf(vu: HvxVector) -> HvxVector {
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
pub unsafe fn q6_v_vgetqfext_vr(vu: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_v_vsetqfext_vr(vu: HvxVector, rt: i32) -> HvxVector {
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
pub unsafe fn q6_v_vabs_v(vu: HvxVector) -> HvxVector {
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
pub unsafe fn q6_whf_vcvt2_vb(vu: HvxVector) -> HvxVectorPair {
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
pub unsafe fn q6_whf_vcvt2_vub(vu: HvxVector) -> HvxVectorPair {
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
pub unsafe fn q6_whf_vcvt_v(vu: HvxVector) -> HvxVectorPair {
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
pub unsafe fn q6_v_vfmax_vv(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_v_vfmin_vv(vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_v_vfneg_v(vu: HvxVector) -> HvxVector {
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
pub unsafe fn q6_q_and_qq(qs: HvxVectorPred, qt: HvxVectorPred) -> HvxVectorPred {
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
pub unsafe fn q6_q_and_qqn(qs: HvxVectorPred, qt: HvxVectorPred) -> HvxVectorPred {
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
pub unsafe fn q6_q_not_q(qs: HvxVectorPred) -> HvxVectorPred {
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
pub unsafe fn q6_q_or_qq(qs: HvxVectorPred, qt: HvxVectorPred) -> HvxVectorPred {
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
pub unsafe fn q6_q_or_qqn(qs: HvxVectorPred, qt: HvxVectorPred) -> HvxVectorPred {
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
pub unsafe fn q6_q_vsetq_r(rt: i32) -> HvxVectorPred {
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
pub unsafe fn q6_q_xor_qq(qs: HvxVectorPred, qt: HvxVectorPred) -> HvxVectorPred {
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
pub unsafe fn q6_vmem_qnriv(qv: HvxVectorPred, rt: *mut HvxVector, vs: HvxVector) {
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
pub unsafe fn q6_vmem_qnriv_nt(qv: HvxVectorPred, rt: *mut HvxVector, vs: HvxVector) {
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
pub unsafe fn q6_vmem_qriv_nt(qv: HvxVectorPred, rt: *mut HvxVector, vs: HvxVector) {
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
pub unsafe fn q6_vmem_qriv(qv: HvxVectorPred, rt: *mut HvxVector, vs: HvxVector) {
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
pub unsafe fn q6_vb_condacc_qnvbvb(qv: HvxVectorPred, vx: HvxVector, vu: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vb_condacc_qvbvb(qv: HvxVectorPred, vx: HvxVector, vu: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vh_condacc_qnvhvh(qv: HvxVectorPred, vx: HvxVector, vu: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vh_condacc_qvhvh(qv: HvxVectorPred, vx: HvxVector, vu: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vw_condacc_qnvwvw(qv: HvxVectorPred, vx: HvxVector, vu: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vw_condacc_qvwvw(qv: HvxVectorPred, vx: HvxVector, vu: HvxVector) -> HvxVector {
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
pub unsafe fn q6_v_vand_qr(qu: HvxVectorPred, rt: i32) -> HvxVector {
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
pub unsafe fn q6_v_vandor_vqr(vx: HvxVector, qu: HvxVectorPred, rt: i32) -> HvxVector {
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
pub unsafe fn q6_q_vand_vr(vu: HvxVector, rt: i32) -> HvxVectorPred {
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
pub unsafe fn q6_q_vandor_qvr(qx: HvxVectorPred, vu: HvxVector, rt: i32) -> HvxVectorPred {
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
pub unsafe fn q6_q_vcmp_eq_vbvb(vu: HvxVector, vv: HvxVector) -> HvxVectorPred {
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
pub unsafe fn q6_q_vcmp_eqand_qvbvb(
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
pub unsafe fn q6_q_vcmp_eqor_qvbvb(
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
pub unsafe fn q6_q_vcmp_eqxacc_qvbvb(
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
pub unsafe fn q6_q_vcmp_eq_vhvh(vu: HvxVector, vv: HvxVector) -> HvxVectorPred {
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
pub unsafe fn q6_q_vcmp_eqand_qvhvh(
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
pub unsafe fn q6_q_vcmp_eqor_qvhvh(
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
pub unsafe fn q6_q_vcmp_eqxacc_qvhvh(
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
pub unsafe fn q6_q_vcmp_eq_vwvw(vu: HvxVector, vv: HvxVector) -> HvxVectorPred {
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
pub unsafe fn q6_q_vcmp_eqand_qvwvw(
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
pub unsafe fn q6_q_vcmp_eqor_qvwvw(
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
pub unsafe fn q6_q_vcmp_eqxacc_qvwvw(
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
pub unsafe fn q6_q_vcmp_gt_vbvb(vu: HvxVector, vv: HvxVector) -> HvxVectorPred {
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
pub unsafe fn q6_q_vcmp_gtand_qvbvb(
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
pub unsafe fn q6_q_vcmp_gtor_qvbvb(
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
pub unsafe fn q6_q_vcmp_gtxacc_qvbvb(
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
pub unsafe fn q6_q_vcmp_gt_vhvh(vu: HvxVector, vv: HvxVector) -> HvxVectorPred {
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
pub unsafe fn q6_q_vcmp_gtand_qvhvh(
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
pub unsafe fn q6_q_vcmp_gtor_qvhvh(
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
pub unsafe fn q6_q_vcmp_gtxacc_qvhvh(
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
pub unsafe fn q6_q_vcmp_gt_vubvub(vu: HvxVector, vv: HvxVector) -> HvxVectorPred {
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
pub unsafe fn q6_q_vcmp_gtand_qvubvub(
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
pub unsafe fn q6_q_vcmp_gtor_qvubvub(
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
pub unsafe fn q6_q_vcmp_gtxacc_qvubvub(
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
pub unsafe fn q6_q_vcmp_gt_vuhvuh(vu: HvxVector, vv: HvxVector) -> HvxVectorPred {
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
pub unsafe fn q6_q_vcmp_gtand_qvuhvuh(
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
pub unsafe fn q6_q_vcmp_gtor_qvuhvuh(
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
pub unsafe fn q6_q_vcmp_gtxacc_qvuhvuh(
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
pub unsafe fn q6_q_vcmp_gt_vuwvuw(vu: HvxVector, vv: HvxVector) -> HvxVectorPred {
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
pub unsafe fn q6_q_vcmp_gtand_qvuwvuw(
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
pub unsafe fn q6_q_vcmp_gtor_qvuwvuw(
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
pub unsafe fn q6_q_vcmp_gtxacc_qvuwvuw(
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
pub unsafe fn q6_q_vcmp_gt_vwvw(vu: HvxVector, vv: HvxVector) -> HvxVectorPred {
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
pub unsafe fn q6_q_vcmp_gtand_qvwvw(
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
pub unsafe fn q6_q_vcmp_gtor_qvwvw(
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
pub unsafe fn q6_q_vcmp_gtxacc_qvwvw(
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
pub unsafe fn q6_v_vmux_qvv(qt: HvxVectorPred, vu: HvxVector, vv: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vb_condnac_qnvbvb(qv: HvxVectorPred, vx: HvxVector, vu: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vb_condnac_qvbvb(qv: HvxVectorPred, vx: HvxVector, vu: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vh_condnac_qnvhvh(qv: HvxVectorPred, vx: HvxVector, vu: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vh_condnac_qvhvh(qv: HvxVectorPred, vx: HvxVector, vu: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vw_condnac_qnvwvw(qv: HvxVectorPred, vx: HvxVector, vu: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vw_condnac_qvwvw(qv: HvxVectorPred, vx: HvxVector, vu: HvxVector) -> HvxVector {
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
pub unsafe fn q6_w_vswap_qvv(qt: HvxVectorPred, vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
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
pub unsafe fn q6_q_vsetq2_r(rt: i32) -> HvxVectorPred {
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
pub unsafe fn q6_qb_vshuffe_qhqh(qs: HvxVectorPred, qt: HvxVectorPred) -> HvxVectorPred {
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
pub unsafe fn q6_qh_vshuffe_qwqw(qs: HvxVectorPred, qt: HvxVectorPred) -> HvxVectorPred {
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
pub unsafe fn q6_v_vand_qnr(qu: HvxVectorPred, rt: i32) -> HvxVector {
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
pub unsafe fn q6_v_vandor_vqnr(vx: HvxVector, qu: HvxVectorPred, rt: i32) -> HvxVector {
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
pub unsafe fn q6_v_vand_qnv(qv: HvxVectorPred, vu: HvxVector) -> HvxVector {
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
pub unsafe fn q6_v_vand_qv(qv: HvxVectorPred, vu: HvxVector) -> HvxVector {
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
pub unsafe fn q6_vgather_aqrmvh(
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
pub unsafe fn q6_vgather_aqrmww(
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
pub unsafe fn q6_vgather_aqrmvw(
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
pub unsafe fn q6_vb_prefixsum_q(qv: HvxVectorPred) -> HvxVector {
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
pub unsafe fn q6_vh_prefixsum_q(qv: HvxVectorPred) -> HvxVector {
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
pub unsafe fn q6_vw_prefixsum_q(qv: HvxVectorPred) -> HvxVector {
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
pub unsafe fn q6_vscatter_qrmvhv(
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
pub unsafe fn q6_vscatter_qrmwwv(
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
pub unsafe fn q6_vscatter_qrmvwv(
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
pub unsafe fn q6_vw_vadd_vwvwq_carry_sat(
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
pub unsafe fn q6_q_vcmp_gt_vhfvhf(vu: HvxVector, vv: HvxVector) -> HvxVectorPred {
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
pub unsafe fn q6_q_vcmp_gtand_qvhfvhf(
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
pub unsafe fn q6_q_vcmp_gtor_qvhfvhf(
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
pub unsafe fn q6_q_vcmp_gtxacc_qvhfvhf(
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
pub unsafe fn q6_q_vcmp_gt_vsfvsf(vu: HvxVector, vv: HvxVector) -> HvxVectorPred {
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
pub unsafe fn q6_q_vcmp_gtand_qvsfvsf(
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
pub unsafe fn q6_q_vcmp_gtor_qvsfvsf(
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
pub unsafe fn q6_q_vcmp_gtxacc_qvsfvsf(
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
