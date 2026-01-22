//! Hexagon HVX intrinsics
//!
//! This module provides intrinsics for the Hexagon Vector Extensions (HVX).
//! HVX is a wide vector extension designed for high-performance signal processing.
//! [Hexagon HVX Programmer's Reference Manual](https://docs.qualcomm.com/doc/80-N2040-61)
//!
//! ## Vector Types
//!
//! HVX supports different vector lengths depending on the configuration:
//! - 128-byte mode: `HvxVector` is 1024 bits (128 bytes)
//! - 64-byte mode: `HvxVector` is 512 bits (64 bytes)
//!
//! This implementation targets 128-byte mode by default. To change the vector
//! length mode, use the appropriate target feature when compiling:
//! - For 128-byte mode: `-C target-feature=+hvx-length128b`
//! - For 64-byte mode: `-C target-feature=+hvx-length64b`
//!
//! Note that HVX v66 and later default to 128-byte mode, while earlier versions
//! default to 64-byte mode.
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
//! - HVX v81: `-C target-feature=+hvxv81`
//!
//! Each version includes all features from previous versions.

#![allow(non_camel_case_types)]

types! {
    #![unstable(feature = "stdarch_hexagon", issue = "none")]

    /// HVX vector type (1024 bits / 128 bytes for 128-byte mode)
    ///
    /// This type represents a single HVX vector register containing 32 x 32-bit values.
    pub struct HvxVector(32 x i32);

    /// HVX vector pair type (2048 bits / 256 bytes for 128-byte mode)
    ///
    /// This type represents a pair of HVX vector registers, often used for
    /// operations that produce double-width results.
    pub struct HvxVectorPair(64 x i32);

    /// HVX vector predicate type (1024 bits / 128 bytes for 128-byte mode)
    ///
    /// This type represents a predicate vector used for conditional operations.
    /// Each bit corresponds to a lane in the vector.
    pub struct HvxVectorPred(32 x i32);
}

// LLVM intrinsic declarations
#[allow(improper_ctypes)]
unsafe extern "unadjusted" {
    #[link_name = "llvm.hexagon.V6.extractw"]
    fn extractw(_: HvxVector, _: i32) -> i32;
    #[link_name = "llvm.hexagon.V6.get_qfext"]
    fn get_qfext(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.get_qfext_oracc"]
    fn get_qfext_oracc(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
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
    #[link_name = "llvm.hexagon.V6.pred_and"]
    fn pred_and(_: HvxVectorPred, _: HvxVectorPred) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.pred_and_n"]
    fn pred_and_n(_: HvxVectorPred, _: HvxVectorPred) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.pred_not"]
    fn pred_not(_: HvxVectorPred) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.pred_or"]
    fn pred_or(_: HvxVectorPred, _: HvxVectorPred) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.pred_or_n"]
    fn pred_or_n(_: HvxVectorPred, _: HvxVectorPred) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.pred_scalar2"]
    fn pred_scalar2(_: i32) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.pred_scalar2v2"]
    fn pred_scalar2v2(_: i32) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.pred_xor"]
    fn pred_xor(_: HvxVectorPred, _: HvxVectorPred) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.set_qfext"]
    fn set_qfext(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.shuffeqh"]
    fn shuffeqh(_: HvxVectorPred, _: HvxVectorPred) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.shuffeqw"]
    fn shuffeqw(_: HvxVectorPred, _: HvxVectorPred) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.v6mpyhubs10"]
    fn v6mpyhubs10(_: HvxVectorPair, _: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.v6mpyhubs10_vxx"]
    fn v6mpyhubs10_vxx(
        _: HvxVectorPair,
        _: HvxVectorPair,
        _: HvxVectorPair,
        _: i32,
    ) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.v6mpyvubs10"]
    fn v6mpyvubs10(_: HvxVectorPair, _: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.v6mpyvubs10_vxx"]
    fn v6mpyvubs10_vxx(
        _: HvxVectorPair,
        _: HvxVectorPair,
        _: HvxVectorPair,
        _: i32,
    ) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vS32b_nqpred_ai"]
    fn vS32b_nqpred_ai(_: HvxVectorPred, _: *mut HvxVector, _: HvxVector) -> ();
    #[link_name = "llvm.hexagon.V6.vS32b_nt_nqpred_ai"]
    fn vS32b_nt_nqpred_ai(_: HvxVectorPred, _: *mut HvxVector, _: HvxVector) -> ();
    #[link_name = "llvm.hexagon.V6.vS32b_nt_qpred_ai"]
    fn vS32b_nt_qpred_ai(_: HvxVectorPred, _: *mut HvxVector, _: HvxVector) -> ();
    #[link_name = "llvm.hexagon.V6.vS32b_qpred_ai"]
    fn vS32b_qpred_ai(_: HvxVectorPred, _: *mut HvxVector, _: HvxVector) -> ();
    #[link_name = "llvm.hexagon.V6.vabs_f8"]
    fn vabs_f8(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vabs_hf"]
    fn vabs_hf(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vabs_qf16_hf"]
    fn vabs_qf16_hf(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vabs_qf16_qf16"]
    fn vabs_qf16_qf16(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vabs_qf32_qf32"]
    fn vabs_qf32_qf32(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vabs_qf32_sf"]
    fn vabs_qf32_sf(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vabs_sf"]
    fn vabs_sf(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vabsb"]
    fn vabsb(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vabsb_sat"]
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
    #[link_name = "llvm.hexagon.V6.vabsh_sat"]
    fn vabsh_sat(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vabsw"]
    fn vabsw(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vabsw_sat"]
    fn vabsw_sat(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vadd_hf"]
    fn vadd_hf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vadd_hf_f8"]
    fn vadd_hf_f8(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vadd_hf_hf"]
    fn vadd_hf_hf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vadd_qf16"]
    fn vadd_qf16(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vadd_qf16_mix"]
    fn vadd_qf16_mix(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vadd_qf32"]
    fn vadd_qf32(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vadd_qf32_mix"]
    fn vadd_qf32_mix(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vadd_sf"]
    fn vadd_sf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vadd_sf_bf"]
    fn vadd_sf_bf(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vadd_sf_hf"]
    fn vadd_sf_hf(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vadd_sf_sf"]
    fn vadd_sf_sf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaddb"]
    fn vaddb(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaddb_dv"]
    fn vaddb_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vaddbnq"]
    fn vaddbnq(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaddbq"]
    fn vaddbq(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaddbsat"]
    fn vaddbsat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaddbsat_dv"]
    fn vaddbsat_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vaddcarry"]
    fn vaddcarry(_: HvxVector, _: HvxVector, _: *mut HvxVectorPred) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaddcarrysat"]
    fn vaddcarrysat(_: HvxVector, _: HvxVector, _: HvxVectorPred) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaddclbh"]
    fn vaddclbh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaddclbw"]
    fn vaddclbw(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaddh"]
    fn vaddh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaddh_dv"]
    fn vaddh_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vaddhnq"]
    fn vaddhnq(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaddhq"]
    fn vaddhq(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaddhsat"]
    fn vaddhsat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaddhsat_dv"]
    fn vaddhsat_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vaddhw"]
    fn vaddhw(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vaddhw_acc"]
    fn vaddhw_acc(_: HvxVectorPair, _: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vaddubh"]
    fn vaddubh(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vaddubh_acc"]
    fn vaddubh_acc(_: HvxVectorPair, _: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vaddubsat"]
    fn vaddubsat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaddubsat_dv"]
    fn vaddubsat_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vaddububb_sat"]
    fn vaddububb_sat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vadduhsat"]
    fn vadduhsat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vadduhsat_dv"]
    fn vadduhsat_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vadduhw"]
    fn vadduhw(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vadduhw_acc"]
    fn vadduhw_acc(_: HvxVectorPair, _: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vadduwsat"]
    fn vadduwsat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vadduwsat_dv"]
    fn vadduwsat_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vaddw"]
    fn vaddw(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaddw_dv"]
    fn vaddw_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vaddwnq"]
    fn vaddwnq(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaddwq"]
    fn vaddwq(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaddwsat"]
    fn vaddwsat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaddwsat_dv"]
    fn vaddwsat_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.valign4"]
    fn valign4(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.valignb"]
    fn valignb(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.valignbi"]
    fn valignbi(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vand"]
    fn vand(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vandnqrt"]
    fn vandnqrt(_: HvxVectorPred, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vandnqrt_acc"]
    fn vandnqrt_acc(_: HvxVector, _: HvxVectorPred, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vandqrt"]
    fn vandqrt(_: HvxVectorPred, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vandqrt_acc"]
    fn vandqrt_acc(_: HvxVector, _: HvxVectorPred, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vandvnqv"]
    fn vandvnqv(_: HvxVectorPred, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vandvqv"]
    fn vandvqv(_: HvxVectorPred, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaslh"]
    fn vaslh(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaslh_acc"]
    fn vaslh_acc(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaslhv"]
    fn vaslhv(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaslw"]
    fn vaslw(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaslw_acc"]
    fn vaslw_acc(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vaslwv"]
    fn vaslwv(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vasr_into"]
    fn vasr_into(_: HvxVectorPair, _: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vasrh"]
    fn vasrh(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vasrh_acc"]
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
    #[link_name = "llvm.hexagon.V6.vasrw_acc"]
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
    #[link_name = "llvm.hexagon.V6.vassign_fp"]
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
    #[link_name = "llvm.hexagon.V6.vconv_bf_qf32"]
    fn vconv_bf_qf32(_: HvxVectorPair) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vconv_f8_qf16"]
    fn vconv_f8_qf16(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vconv_h_hf"]
    fn vconv_h_hf(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vconv_h_hf_rnd"]
    fn vconv_h_hf_rnd(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vconv_hf_h"]
    fn vconv_hf_h(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vconv_hf_qf16"]
    fn vconv_hf_qf16(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vconv_hf_qf32"]
    fn vconv_hf_qf32(_: HvxVectorPair) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vconv_qf16_f8"]
    fn vconv_qf16_f8(_: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vconv_qf16_hf"]
    fn vconv_qf16_hf(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vconv_qf16_qf16"]
    fn vconv_qf16_qf16(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vconv_qf32_qf32"]
    fn vconv_qf32_qf32(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vconv_qf32_sf"]
    fn vconv_qf32_sf(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vconv_sf_qf32"]
    fn vconv_sf_qf32(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vconv_sf_w"]
    fn vconv_sf_w(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vconv_w_sf"]
    fn vconv_w_sf(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vcvt2_b_hf"]
    fn vcvt2_b_hf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vcvt2_hf_b"]
    fn vcvt2_hf_b(_: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vcvt2_hf_ub"]
    fn vcvt2_hf_ub(_: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vcvt2_ub_hf"]
    fn vcvt2_ub_hf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vcvt_b_hf"]
    fn vcvt_b_hf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vcvt_bf_sf"]
    fn vcvt_bf_sf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vcvt_f8_hf"]
    fn vcvt_f8_hf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vcvt_h_hf"]
    fn vcvt_h_hf(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vcvt_hf_b"]
    fn vcvt_hf_b(_: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vcvt_hf_f8"]
    fn vcvt_hf_f8(_: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vcvt_hf_h"]
    fn vcvt_hf_h(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vcvt_hf_sf"]
    fn vcvt_hf_sf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vcvt_hf_ub"]
    fn vcvt_hf_ub(_: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vcvt_hf_uh"]
    fn vcvt_hf_uh(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vcvt_sf_hf"]
    fn vcvt_sf_hf(_: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vcvt_ub_hf"]
    fn vcvt_ub_hf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vcvt_uh_hf"]
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
    #[link_name = "llvm.hexagon.V6.vdmpy_sf_hf"]
    fn vdmpy_sf_hf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdmpy_sf_hf_acc"]
    fn vdmpy_sf_hf_acc(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdmpybus"]
    fn vdmpybus(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdmpybus_acc"]
    fn vdmpybus_acc(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdmpybus_dv"]
    fn vdmpybus_dv(_: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vdmpybus_dv_acc"]
    fn vdmpybus_dv_acc(_: HvxVectorPair, _: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vdmpyhb"]
    fn vdmpyhb(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdmpyhb_acc"]
    fn vdmpyhb_acc(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdmpyhb_dv"]
    fn vdmpyhb_dv(_: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vdmpyhb_dv_acc"]
    fn vdmpyhb_dv_acc(_: HvxVectorPair, _: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vdmpyhisat"]
    fn vdmpyhisat(_: HvxVectorPair, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdmpyhisat_acc"]
    fn vdmpyhisat_acc(_: HvxVector, _: HvxVectorPair, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdmpyhsat"]
    fn vdmpyhsat(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdmpyhsat_acc"]
    fn vdmpyhsat_acc(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdmpyhsuisat"]
    fn vdmpyhsuisat(_: HvxVectorPair, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdmpyhsuisat_acc"]
    fn vdmpyhsuisat_acc(_: HvxVector, _: HvxVectorPair, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdmpyhsusat"]
    fn vdmpyhsusat(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdmpyhsusat_acc"]
    fn vdmpyhsusat_acc(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdmpyhvsat"]
    fn vdmpyhvsat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdmpyhvsat_acc"]
    fn vdmpyhvsat_acc(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vdsaduh"]
    fn vdsaduh(_: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vdsaduh_acc"]
    fn vdsaduh_acc(_: HvxVectorPair, _: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.veqb"]
    fn veqb(_: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.veqb_and"]
    fn veqb_and(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.veqb_or"]
    fn veqb_or(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.veqb_xor"]
    fn veqb_xor(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.veqh"]
    fn veqh(_: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.veqh_and"]
    fn veqh_and(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.veqh_or"]
    fn veqh_or(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.veqh_xor"]
    fn veqh_xor(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.veqhf"]
    fn veqhf(_: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.veqhf_and"]
    fn veqhf_and(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.veqhf_or"]
    fn veqhf_or(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.veqhf_xor"]
    fn veqhf_xor(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.veqsf"]
    fn veqsf(_: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.veqsf_and"]
    fn veqsf_and(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.veqsf_or"]
    fn veqsf_or(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.veqsf_xor"]
    fn veqsf_xor(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.veqw"]
    fn veqw(_: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.veqw_and"]
    fn veqw_and(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.veqw_or"]
    fn veqw_or(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.veqw_xor"]
    fn veqw_xor(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.vfmax_f8"]
    fn vfmax_f8(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vfmax_hf"]
    fn vfmax_hf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vfmax_sf"]
    fn vfmax_sf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vfmin_f8"]
    fn vfmin_f8(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vfmin_hf"]
    fn vfmin_hf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vfmin_sf"]
    fn vfmin_sf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vfneg_f8"]
    fn vfneg_f8(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vfneg_hf"]
    fn vfneg_hf(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vfneg_sf"]
    fn vfneg_sf(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vgathermh"]
    fn vgathermh(_: *mut HvxVector, _: i32, _: i32, _: HvxVector) -> ();
    #[link_name = "llvm.hexagon.V6.vgathermhq"]
    fn vgathermhq(_: *mut HvxVector, _: HvxVectorPred, _: i32, _: i32, _: HvxVector) -> ();
    #[link_name = "llvm.hexagon.V6.vgathermhw"]
    fn vgathermhw(_: *mut HvxVector, _: i32, _: i32, _: HvxVectorPair) -> ();
    #[link_name = "llvm.hexagon.V6.vgathermhwq"]
    fn vgathermhwq(_: *mut HvxVector, _: HvxVectorPred, _: i32, _: i32, _: HvxVectorPair) -> ();
    #[link_name = "llvm.hexagon.V6.vgathermw"]
    fn vgathermw(_: *mut HvxVector, _: i32, _: i32, _: HvxVector) -> ();
    #[link_name = "llvm.hexagon.V6.vgathermwq"]
    fn vgathermwq(_: *mut HvxVector, _: HvxVectorPred, _: i32, _: i32, _: HvxVector) -> ();
    #[link_name = "llvm.hexagon.V6.vgtb"]
    fn vgtb(_: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.vgtb_and"]
    fn vgtb_and(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.vgtb_or"]
    fn vgtb_or(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.vgtb_xor"]
    fn vgtb_xor(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.vgtbf"]
    fn vgtbf(_: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.vgtbf_and"]
    fn vgtbf_and(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.vgtbf_or"]
    fn vgtbf_or(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.vgtbf_xor"]
    fn vgtbf_xor(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.vgth"]
    fn vgth(_: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.vgth_and"]
    fn vgth_and(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.vgth_or"]
    fn vgth_or(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.vgth_xor"]
    fn vgth_xor(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.vgthf"]
    fn vgthf(_: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.vgthf_and"]
    fn vgthf_and(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.vgthf_or"]
    fn vgthf_or(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.vgthf_xor"]
    fn vgthf_xor(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.vgtsf"]
    fn vgtsf(_: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.vgtsf_and"]
    fn vgtsf_and(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.vgtsf_or"]
    fn vgtsf_or(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.vgtsf_xor"]
    fn vgtsf_xor(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.vgtub"]
    fn vgtub(_: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.vgtub_and"]
    fn vgtub_and(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.vgtub_or"]
    fn vgtub_or(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.vgtub_xor"]
    fn vgtub_xor(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.vgtuh"]
    fn vgtuh(_: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.vgtuh_and"]
    fn vgtuh_and(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.vgtuh_or"]
    fn vgtuh_or(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.vgtuh_xor"]
    fn vgtuh_xor(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.vgtuw"]
    fn vgtuw(_: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.vgtuw_and"]
    fn vgtuw_and(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.vgtuw_or"]
    fn vgtuw_or(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.vgtuw_xor"]
    fn vgtuw_xor(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.vgtw"]
    fn vgtw(_: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.vgtw_and"]
    fn vgtw_and(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.vgtw_or"]
    fn vgtw_or(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.vgtw_xor"]
    fn vgtw_xor(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVectorPred;
    #[link_name = "llvm.hexagon.V6.vilog2_hf"]
    fn vilog2_hf(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vilog2_qf16"]
    fn vilog2_qf16(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vilog2_qf32"]
    fn vilog2_qf32(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vilog2_sf"]
    fn vilog2_sf(_: HvxVector) -> HvxVector;
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
    #[link_name = "llvm.hexagon.V6.vlut4"]
    fn vlut4(_: HvxVector, _: i64) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vlutvvb"]
    fn vlutvvb(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vlutvvb_nm"]
    fn vlutvvb_nm(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vlutvvb_oracc"]
    fn vlutvvb_oracc(_: HvxVector, _: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vlutvvb_oracci"]
    fn vlutvvb_oracci(_: HvxVector, _: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vlutvvbi"]
    fn vlutvvbi(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vlutvwh"]
    fn vlutvwh(_: HvxVector, _: HvxVector, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vlutvwh_nm"]
    fn vlutvwh_nm(_: HvxVector, _: HvxVector, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vlutvwh_oracc"]
    fn vlutvwh_oracc(_: HvxVectorPair, _: HvxVector, _: HvxVector, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vlutvwh_oracci"]
    fn vlutvwh_oracci(_: HvxVectorPair, _: HvxVector, _: HvxVector, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vlutvwhi"]
    fn vlutvwhi(_: HvxVector, _: HvxVector, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmax_bf"]
    fn vmax_bf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmax_hf"]
    fn vmax_hf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmax_sf"]
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
    #[link_name = "llvm.hexagon.V6.vmerge_qf"]
    fn vmerge_qf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmin_bf"]
    fn vmin_bf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmin_hf"]
    fn vmin_hf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmin_sf"]
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
    #[link_name = "llvm.hexagon.V6.vmpabus_acc"]
    fn vmpabus_acc(_: HvxVectorPair, _: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpabusv"]
    fn vmpabusv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpabuu"]
    fn vmpabuu(_: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpabuu_acc"]
    fn vmpabuu_acc(_: HvxVectorPair, _: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpabuuv"]
    fn vmpabuuv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpahb"]
    fn vmpahb(_: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpahb_acc"]
    fn vmpahb_acc(_: HvxVectorPair, _: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpahhsat"]
    fn vmpahhsat(_: HvxVector, _: HvxVector, _: i64) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpauhb"]
    fn vmpauhb(_: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpauhb_acc"]
    fn vmpauhb_acc(_: HvxVectorPair, _: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpauhuhsat"]
    fn vmpauhuhsat(_: HvxVector, _: HvxVector, _: i64) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpsuhuhsat"]
    fn vmpsuhuhsat(_: HvxVector, _: HvxVector, _: i64) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpy_hf_f8"]
    fn vmpy_hf_f8(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpy_hf_f8_acc"]
    fn vmpy_hf_f8_acc(_: HvxVectorPair, _: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpy_hf_hf"]
    fn vmpy_hf_hf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpy_hf_hf_acc"]
    fn vmpy_hf_hf_acc(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpy_qf16"]
    fn vmpy_qf16(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpy_qf16_hf"]
    fn vmpy_qf16_hf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpy_qf16_mix_hf"]
    fn vmpy_qf16_mix_hf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpy_qf32"]
    fn vmpy_qf32(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpy_qf32_hf"]
    fn vmpy_qf32_hf(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpy_qf32_mix_hf"]
    fn vmpy_qf32_mix_hf(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpy_qf32_qf16"]
    fn vmpy_qf32_qf16(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpy_qf32_sf"]
    fn vmpy_qf32_sf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpy_rt_hf"]
    fn vmpy_rt_hf(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpy_rt_qf16"]
    fn vmpy_rt_qf16(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpy_rt_sf"]
    fn vmpy_rt_sf(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpy_sf_bf"]
    fn vmpy_sf_bf(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpy_sf_bf_acc"]
    fn vmpy_sf_bf_acc(_: HvxVectorPair, _: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpy_sf_hf"]
    fn vmpy_sf_hf(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpy_sf_hf_acc"]
    fn vmpy_sf_hf_acc(_: HvxVectorPair, _: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpy_sf_sf"]
    fn vmpy_sf_sf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpybus"]
    fn vmpybus(_: HvxVector, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpybus_acc"]
    fn vmpybus_acc(_: HvxVectorPair, _: HvxVector, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpybusv"]
    fn vmpybusv(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpybusv_acc"]
    fn vmpybusv_acc(_: HvxVectorPair, _: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpybv"]
    fn vmpybv(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpybv_acc"]
    fn vmpybv_acc(_: HvxVectorPair, _: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyewuh"]
    fn vmpyewuh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyewuh_64"]
    fn vmpyewuh_64(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyh"]
    fn vmpyh(_: HvxVector, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyh_acc"]
    fn vmpyh_acc(_: HvxVectorPair, _: HvxVector, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyhsat_acc"]
    fn vmpyhsat_acc(_: HvxVectorPair, _: HvxVector, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyhsrs"]
    fn vmpyhsrs(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyhss"]
    fn vmpyhss(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyhus"]
    fn vmpyhus(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyhus_acc"]
    fn vmpyhus_acc(_: HvxVectorPair, _: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyhv"]
    fn vmpyhv(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyhv_acc"]
    fn vmpyhv_acc(_: HvxVectorPair, _: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyhvsrs"]
    fn vmpyhvsrs(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyieoh"]
    fn vmpyieoh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyiewh_acc"]
    fn vmpyiewh_acc(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyiewuh"]
    fn vmpyiewuh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyiewuh_acc"]
    fn vmpyiewuh_acc(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyih"]
    fn vmpyih(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyih_acc"]
    fn vmpyih_acc(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyihb"]
    fn vmpyihb(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyihb_acc"]
    fn vmpyihb_acc(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyiowh"]
    fn vmpyiowh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyiwb"]
    fn vmpyiwb(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyiwb_acc"]
    fn vmpyiwb_acc(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyiwh"]
    fn vmpyiwh(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyiwh_acc"]
    fn vmpyiwh_acc(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyiwub"]
    fn vmpyiwub(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyiwub_acc"]
    fn vmpyiwub_acc(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyowh"]
    fn vmpyowh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyowh_64_acc"]
    fn vmpyowh_64_acc(_: HvxVectorPair, _: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyowh_rnd"]
    fn vmpyowh_rnd(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyowh_rnd_sacc"]
    fn vmpyowh_rnd_sacc(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyowh_sacc"]
    fn vmpyowh_sacc(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyub"]
    fn vmpyub(_: HvxVector, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyub_acc"]
    fn vmpyub_acc(_: HvxVectorPair, _: HvxVector, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyubv"]
    fn vmpyubv(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyubv_acc"]
    fn vmpyubv_acc(_: HvxVectorPair, _: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyuh"]
    fn vmpyuh(_: HvxVector, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyuh_acc"]
    fn vmpyuh_acc(_: HvxVectorPair, _: HvxVector, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyuhe"]
    fn vmpyuhe(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyuhe_acc"]
    fn vmpyuhe_acc(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmpyuhv"]
    fn vmpyuhv(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyuhv_acc"]
    fn vmpyuhv_acc(_: HvxVectorPair, _: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vmpyuhvs"]
    fn vmpyuhvs(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vmux"]
    fn vmux(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vnavgb"]
    fn vnavgb(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vnavgh"]
    fn vnavgh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vnavgub"]
    fn vnavgub(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vnavgw"]
    fn vnavgw(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vneg_qf16_hf"]
    fn vneg_qf16_hf(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vneg_qf16_qf16"]
    fn vneg_qf16_qf16(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vneg_qf32_qf32"]
    fn vneg_qf32_qf32(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vneg_qf32_sf"]
    fn vneg_qf32_sf(_: HvxVector) -> HvxVector;
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
    #[link_name = "llvm.hexagon.V6.vpackhb_sat"]
    fn vpackhb_sat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vpackhub_sat"]
    fn vpackhub_sat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vpackob"]
    fn vpackob(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vpackoh"]
    fn vpackoh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vpackwh_sat"]
    fn vpackwh_sat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vpackwuh_sat"]
    fn vpackwuh_sat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vpopcounth"]
    fn vpopcounth(_: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vprefixqb"]
    fn vprefixqb(_: HvxVectorPred) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vprefixqh"]
    fn vprefixqh(_: HvxVectorPred) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vprefixqw"]
    fn vprefixqw(_: HvxVectorPred) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vrdelta"]
    fn vrdelta(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vrmpybus"]
    fn vrmpybus(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vrmpybus_acc"]
    fn vrmpybus_acc(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vrmpybusi"]
    fn vrmpybusi(_: HvxVectorPair, _: i32, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vrmpybusi_acc"]
    fn vrmpybusi_acc(_: HvxVectorPair, _: HvxVectorPair, _: i32, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vrmpybusv"]
    fn vrmpybusv(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vrmpybusv_acc"]
    fn vrmpybusv_acc(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vrmpybv"]
    fn vrmpybv(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vrmpybv_acc"]
    fn vrmpybv_acc(_: HvxVector, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vrmpyub"]
    fn vrmpyub(_: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vrmpyub_acc"]
    fn vrmpyub_acc(_: HvxVector, _: HvxVector, _: i32) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vrmpyubi"]
    fn vrmpyubi(_: HvxVectorPair, _: i32, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vrmpyubi_acc"]
    fn vrmpyubi_acc(_: HvxVectorPair, _: HvxVectorPair, _: i32, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vrmpyubv"]
    fn vrmpyubv(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vrmpyubv_acc"]
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
    #[link_name = "llvm.hexagon.V6.vrsadubi_acc"]
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
    #[link_name = "llvm.hexagon.V6.vscattermh_add"]
    fn vscattermh_add(_: i32, _: i32, _: HvxVector, _: HvxVector) -> ();
    #[link_name = "llvm.hexagon.V6.vscattermhq"]
    fn vscattermhq(_: HvxVectorPred, _: i32, _: i32, _: HvxVector, _: HvxVector) -> ();
    #[link_name = "llvm.hexagon.V6.vscattermhw"]
    fn vscattermhw(_: i32, _: i32, _: HvxVectorPair, _: HvxVector) -> ();
    #[link_name = "llvm.hexagon.V6.vscattermhw_add"]
    fn vscattermhw_add(_: i32, _: i32, _: HvxVectorPair, _: HvxVector) -> ();
    #[link_name = "llvm.hexagon.V6.vscattermhwq"]
    fn vscattermhwq(_: HvxVectorPred, _: i32, _: i32, _: HvxVectorPair, _: HvxVector) -> ();
    #[link_name = "llvm.hexagon.V6.vscattermw"]
    fn vscattermw(_: i32, _: i32, _: HvxVector, _: HvxVector) -> ();
    #[link_name = "llvm.hexagon.V6.vscattermw_add"]
    fn vscattermw_add(_: i32, _: i32, _: HvxVector, _: HvxVector) -> ();
    #[link_name = "llvm.hexagon.V6.vscattermwq"]
    fn vscattermwq(_: HvxVectorPred, _: i32, _: i32, _: HvxVector, _: HvxVector) -> ();
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
    #[link_name = "llvm.hexagon.V6.vsub_hf"]
    fn vsub_hf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsub_hf_f8"]
    fn vsub_hf_f8(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vsub_hf_hf"]
    fn vsub_hf_hf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsub_hf_mix"]
    fn vsub_hf_mix(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsub_qf16"]
    fn vsub_qf16(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsub_qf16_mix"]
    fn vsub_qf16_mix(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsub_qf32"]
    fn vsub_qf32(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsub_qf32_mix"]
    fn vsub_qf32_mix(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsub_sf"]
    fn vsub_sf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsub_sf_bf"]
    fn vsub_sf_bf(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vsub_sf_hf"]
    fn vsub_sf_hf(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vsub_sf_mix"]
    fn vsub_sf_mix(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsub_sf_sf"]
    fn vsub_sf_sf(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsubb"]
    fn vsubb(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsubb_dv"]
    fn vsubb_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vsubbnq"]
    fn vsubbnq(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsubbq"]
    fn vsubbq(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsubbsat"]
    fn vsubbsat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsubbsat_dv"]
    fn vsubbsat_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vsubcarry"]
    fn vsubcarry(_: HvxVector, _: HvxVector, _: *mut HvxVectorPred) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsubh"]
    fn vsubh(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsubh_dv"]
    fn vsubh_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vsubhnq"]
    fn vsubhnq(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsubhq"]
    fn vsubhq(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsubhsat"]
    fn vsubhsat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsubhsat_dv"]
    fn vsubhsat_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vsubhw"]
    fn vsubhw(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vsububh"]
    fn vsububh(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vsububsat"]
    fn vsububsat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsububsat_dv"]
    fn vsububsat_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vsubububb_sat"]
    fn vsubububb_sat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsubuhsat"]
    fn vsubuhsat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsubuhsat_dv"]
    fn vsubuhsat_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vsubuhw"]
    fn vsubuhw(_: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vsubuwsat"]
    fn vsubuwsat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsubuwsat_dv"]
    fn vsubuwsat_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vsubw"]
    fn vsubw(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsubw_dv"]
    fn vsubw_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vsubwnq"]
    fn vsubwnq(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsubwq"]
    fn vsubwq(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsubwsat"]
    fn vsubwsat(_: HvxVector, _: HvxVector) -> HvxVector;
    #[link_name = "llvm.hexagon.V6.vsubwsat_dv"]
    fn vsubwsat_dv(_: HvxVectorPair, _: HvxVectorPair) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vswap"]
    fn vswap(_: HvxVectorPred, _: HvxVector, _: HvxVector) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vtmpyb"]
    fn vtmpyb(_: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vtmpyb_acc"]
    fn vtmpyb_acc(_: HvxVectorPair, _: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vtmpybus"]
    fn vtmpybus(_: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vtmpybus_acc"]
    fn vtmpybus_acc(_: HvxVectorPair, _: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vtmpyhb"]
    fn vtmpyhb(_: HvxVectorPair, _: i32) -> HvxVectorPair;
    #[link_name = "llvm.hexagon.V6.vtmpyhb_acc"]
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

// ============================================================================
// HVX Architecture v60 intrinsics
// Target feature: hvxv60
// ============================================================================

/// `Rd32=vextract(Vu32,Rs32)`
///
/// Instruction Type: LD
/// Execution Slots: SLOT0
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_r_vextract_vr(vu: HvxVector, rs: i32) -> i32 {
    extractw(vu, rs)
}

/// `Vd32=hi(Vss32)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_v_hi_w(vss: HvxVectorPair) -> HvxVector {
    hi(vss)
}

/// `Vd32=lo(Vss32)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_v_lo_w(vss: HvxVectorPair) -> HvxVector {
    lo(vss)
}

/// `Vd32=vsplat(Rt32)`
///
/// Instruction Type: CVI_VX_LATE
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_v_vsplat_r(rt: i32) -> HvxVector {
    lvsplatw(rt)
}

/// `Vd32.uh=vabsdiff(Vu32.h,Vv32.h)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vuh_vabsdiff_vhvh(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vabsdiffh(vu, vv)
}

/// `Vd32.ub=vabsdiff(Vu32.ub,Vv32.ub)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vub_vabsdiff_vubvub(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vabsdiffub(vu, vv)
}

/// `Vd32.uh=vabsdiff(Vu32.uh,Vv32.uh)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vuh_vabsdiff_vuhvuh(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vabsdiffuh(vu, vv)
}

/// `Vd32.uw=vabsdiff(Vu32.w,Vv32.w)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vuw_vabsdiff_vwvw(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vabsdiffw(vu, vv)
}

/// `Vd32.h=vabs(Vu32.h)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_vabs_vh(vu: HvxVector) -> HvxVector {
    vabsh(vu)
}

/// `Vd32.h=vabs(Vu32.h):sat`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_vabs_vh_sat(vu: HvxVector) -> HvxVector {
    vabsh_sat(vu)
}

/// `Vd32.w=vabs(Vu32.w)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vabs_vw(vu: HvxVector) -> HvxVector {
    vabsw(vu)
}

/// `Vd32.w=vabs(Vu32.w):sat`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vabs_vw_sat(vu: HvxVector) -> HvxVector {
    vabsw_sat(vu)
}

/// `Vd32.b=vadd(Vu32.b,Vv32.b)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vb_vadd_vbvb(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vaddb(vu, vv)
}

/// `Vdd32.b=vadd(Vuu32.b,Vvv32.b)`
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wb_vadd_wbwb(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
    vaddb_dv(vuu, vvv)
}

/// `Vd32.h=vadd(Vu32.h,Vv32.h)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_vadd_vhvh(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vaddh(vu, vv)
}

/// `Vdd32.h=vadd(Vuu32.h,Vvv32.h)`
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wh_vadd_whwh(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
    vaddh_dv(vuu, vvv)
}

/// `Vd32.h=vadd(Vu32.h,Vv32.h):sat`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_vadd_vhvh_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vaddhsat(vu, vv)
}

/// `Vdd32.h=vadd(Vuu32.h,Vvv32.h):sat`
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wh_vadd_whwh_sat(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
    vaddhsat_dv(vuu, vvv)
}

/// `Vdd32.w=vadd(Vu32.h,Vv32.h)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_ww_vadd_vhvh(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
    vaddhw(vu, vv)
}

/// `Vdd32.h=vadd(Vu32.ub,Vv32.ub)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wh_vadd_vubvub(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
    vaddubh(vu, vv)
}

/// `Vd32.ub=vadd(Vu32.ub,Vv32.ub):sat`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vub_vadd_vubvub_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vaddubsat(vu, vv)
}

/// `Vdd32.ub=vadd(Vuu32.ub,Vvv32.ub):sat`
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wub_vadd_wubwub_sat(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
    vaddubsat_dv(vuu, vvv)
}

/// `Vd32.uh=vadd(Vu32.uh,Vv32.uh):sat`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vuh_vadd_vuhvuh_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vadduhsat(vu, vv)
}

/// `Vdd32.uh=vadd(Vuu32.uh,Vvv32.uh):sat`
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wuh_vadd_wuhwuh_sat(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
    vadduhsat_dv(vuu, vvv)
}

/// `Vdd32.w=vadd(Vu32.uh,Vv32.uh)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_ww_vadd_vuhvuh(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
    vadduhw(vu, vv)
}

/// `Vd32.w=vadd(Vu32.w,Vv32.w)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vadd_vwvw(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vaddw(vu, vv)
}

/// `Vdd32.w=vadd(Vuu32.w,Vvv32.w)`
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_ww_vadd_wwww(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
    vaddw_dv(vuu, vvv)
}

/// `Vd32.w=vadd(Vu32.w,Vv32.w):sat`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vadd_vwvw_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vaddwsat(vu, vv)
}

/// `Vdd32.w=vadd(Vuu32.w,Vvv32.w):sat`
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_ww_vadd_wwww_sat(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
    vaddwsat_dv(vuu, vvv)
}

/// `Vd32=valign(Vu32,Vv32,Rt8)`
///
/// Instruction Type: CVI_VP
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_v_valign_vvr(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVector {
    valignb(vu, vv, rt)
}

/// `Vd32=valign(Vu32,Vv32,#u3)`
///
/// Instruction Type: CVI_VP
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_v_valign_vvi(vu: HvxVector, vv: HvxVector, iu3: i32) -> HvxVector {
    valignbi(vu, vv, iu3)
}

/// `Vd32=vand(Vu32,Vv32)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_v_vand_vv(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vand(vu, vv)
}

/// `Vd32.h=vasl(Vu32.h,Rt32)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_vasl_vhr(vu: HvxVector, rt: i32) -> HvxVector {
    vaslh(vu, rt)
}

/// `Vd32.h=vasl(Vu32.h,Vv32.h)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_vasl_vhvh(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vaslhv(vu, vv)
}

/// `Vd32.w=vasl(Vu32.w,Rt32)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vasl_vwr(vu: HvxVector, rt: i32) -> HvxVector {
    vaslw(vu, rt)
}

/// `Vx32.w+=vasl(Vu32.w,Rt32)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vaslacc_vwvwr(vx: HvxVector, vu: HvxVector, rt: i32) -> HvxVector {
    vaslw_acc(vx, vu, rt)
}

/// `Vd32.w=vasl(Vu32.w,Vv32.w)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vasl_vwvw(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vaslwv(vu, vv)
}

/// `Vd32.h=vasr(Vu32.h,Rt32)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_vasr_vhr(vu: HvxVector, rt: i32) -> HvxVector {
    vasrh(vu, rt)
}

/// `Vd32.b=vasr(Vu32.h,Vv32.h,Rt8):rnd:sat`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vb_vasr_vhvhr_rnd_sat(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVector {
    vasrhbrndsat(vu, vv, rt)
}

/// `Vd32.ub=vasr(Vu32.h,Vv32.h,Rt8):rnd:sat`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vub_vasr_vhvhr_rnd_sat(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVector {
    vasrhubrndsat(vu, vv, rt)
}

/// `Vd32.ub=vasr(Vu32.h,Vv32.h,Rt8):sat`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vub_vasr_vhvhr_sat(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVector {
    vasrhubsat(vu, vv, rt)
}

/// `Vd32.h=vasr(Vu32.h,Vv32.h)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_vasr_vhvh(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vasrhv(vu, vv)
}

/// `Vd32.w=vasr(Vu32.w,Rt32)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vasr_vwr(vu: HvxVector, rt: i32) -> HvxVector {
    vasrw(vu, rt)
}

/// `Vx32.w+=vasr(Vu32.w,Rt32)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vasracc_vwvwr(vx: HvxVector, vu: HvxVector, rt: i32) -> HvxVector {
    vasrw_acc(vx, vu, rt)
}

/// `Vd32.h=vasr(Vu32.w,Vv32.w,Rt8)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_vasr_vwvwr(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVector {
    vasrwh(vu, vv, rt)
}

/// `Vd32.h=vasr(Vu32.w,Vv32.w,Rt8):rnd:sat`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_vasr_vwvwr_rnd_sat(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVector {
    vasrwhrndsat(vu, vv, rt)
}

/// `Vd32.h=vasr(Vu32.w,Vv32.w,Rt8):sat`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_vasr_vwvwr_sat(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVector {
    vasrwhsat(vu, vv, rt)
}

/// `Vd32.uh=vasr(Vu32.w,Vv32.w,Rt8):sat`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vuh_vasr_vwvwr_sat(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVector {
    vasrwuhsat(vu, vv, rt)
}

/// `Vd32.w=vasr(Vu32.w,Vv32.w)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vasr_vwvw(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vasrwv(vu, vv)
}

/// `Vd32=Vu32`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_v_equals_v(vu: HvxVector) -> HvxVector {
    vassign(vu)
}

/// `Vdd32=Vuu32`
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_w_equals_w(vuu: HvxVectorPair) -> HvxVectorPair {
    vassignp(vuu)
}

/// `Vd32.h=vavg(Vu32.h,Vv32.h)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_vavg_vhvh(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vavgh(vu, vv)
}

/// `Vd32.h=vavg(Vu32.h,Vv32.h):rnd`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_vavg_vhvh_rnd(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vavghrnd(vu, vv)
}

/// `Vd32.ub=vavg(Vu32.ub,Vv32.ub)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vub_vavg_vubvub(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vavgub(vu, vv)
}

/// `Vd32.ub=vavg(Vu32.ub,Vv32.ub):rnd`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vub_vavg_vubvub_rnd(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vavgubrnd(vu, vv)
}

/// `Vd32.uh=vavg(Vu32.uh,Vv32.uh)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vuh_vavg_vuhvuh(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vavguh(vu, vv)
}

/// `Vd32.uh=vavg(Vu32.uh,Vv32.uh):rnd`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vuh_vavg_vuhvuh_rnd(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vavguhrnd(vu, vv)
}

/// `Vd32.w=vavg(Vu32.w,Vv32.w)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vavg_vwvw(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vavgw(vu, vv)
}

/// `Vd32.w=vavg(Vu32.w,Vv32.w):rnd`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vavg_vwvw_rnd(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vavgwrnd(vu, vv)
}

/// `Vd32.uh=vcl0(Vu32.uh)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vuh_vcl0_vuh(vu: HvxVector) -> HvxVector {
    vcl0h(vu)
}

/// `Vd32.uw=vcl0(Vu32.uw)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vuw_vcl0_vuw(vu: HvxVector) -> HvxVector {
    vcl0w(vu)
}

/// `Vdd32=vcombine(Vu32,Vv32)`
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_w_vcombine_vv(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
    vcombine(vu, vv)
}

/// `Vd32=#0`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_v_vzero() -> HvxVector {
    vd0()
}

/// `Vd32.b=vdeal(Vu32.b)`
///
/// Instruction Type: CVI_VP
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vb_vdeal_vb(vu: HvxVector) -> HvxVector {
    vdealb(vu)
}

/// `Vd32.b=vdeale(Vu32.b,Vv32.b)`
///
/// Instruction Type: CVI_VP
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vb_vdeale_vbvb(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vdealb4w(vu, vv)
}

/// `Vd32.h=vdeal(Vu32.h)`
///
/// Instruction Type: CVI_VP
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_vdeal_vh(vu: HvxVector) -> HvxVector {
    vdealh(vu)
}

/// `Vdd32=vdeal(Vu32,Vv32,Rt8)`
///
/// Instruction Type: CVI_VP_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_w_vdeal_vvr(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVectorPair {
    vdealvdd(vu, vv, rt)
}

/// `Vd32=vdelta(Vu32,Vv32)`
///
/// Instruction Type: CVI_VP
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_v_vdelta_vv(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vdelta(vu, vv)
}

/// `Vd32.h=vdmpy(Vu32.ub,Rt32.b)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_vdmpy_vubrb(vu: HvxVector, rt: i32) -> HvxVector {
    vdmpybus(vu, rt)
}

/// `Vx32.h+=vdmpy(Vu32.ub,Rt32.b)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_vdmpyacc_vhvubrb(vx: HvxVector, vu: HvxVector, rt: i32) -> HvxVector {
    vdmpybus_acc(vx, vu, rt)
}

/// `Vdd32.h=vdmpy(Vuu32.ub,Rt32.b)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wh_vdmpy_wubrb(vuu: HvxVectorPair, rt: i32) -> HvxVectorPair {
    vdmpybus_dv(vuu, rt)
}

/// `Vxx32.h+=vdmpy(Vuu32.ub,Rt32.b)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
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
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vdmpy_vhrb(vu: HvxVector, rt: i32) -> HvxVector {
    vdmpyhb(vu, rt)
}

/// `Vx32.w+=vdmpy(Vu32.h,Rt32.b)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vdmpyacc_vwvhrb(vx: HvxVector, vu: HvxVector, rt: i32) -> HvxVector {
    vdmpyhb_acc(vx, vu, rt)
}

/// `Vdd32.w=vdmpy(Vuu32.h,Rt32.b)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_ww_vdmpy_whrb(vuu: HvxVectorPair, rt: i32) -> HvxVectorPair {
    vdmpyhb_dv(vuu, rt)
}

/// `Vxx32.w+=vdmpy(Vuu32.h,Rt32.b)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
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
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vdmpy_whrh_sat(vuu: HvxVectorPair, rt: i32) -> HvxVector {
    vdmpyhisat(vuu, rt)
}

/// `Vx32.w+=vdmpy(Vuu32.h,Rt32.h):sat`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vdmpyacc_vwwhrh_sat(vx: HvxVector, vuu: HvxVectorPair, rt: i32) -> HvxVector {
    vdmpyhisat_acc(vx, vuu, rt)
}

/// `Vd32.w=vdmpy(Vu32.h,Rt32.h):sat`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vdmpy_vhrh_sat(vu: HvxVector, rt: i32) -> HvxVector {
    vdmpyhsat(vu, rt)
}

/// `Vx32.w+=vdmpy(Vu32.h,Rt32.h):sat`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vdmpyacc_vwvhrh_sat(vx: HvxVector, vu: HvxVector, rt: i32) -> HvxVector {
    vdmpyhsat_acc(vx, vu, rt)
}

/// `Vd32.w=vdmpy(Vuu32.h,Rt32.uh,#1):sat`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vdmpy_whruh_sat(vuu: HvxVectorPair, rt: i32) -> HvxVector {
    vdmpyhsuisat(vuu, rt)
}

/// `Vx32.w+=vdmpy(Vuu32.h,Rt32.uh,#1):sat`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vdmpyacc_vwwhruh_sat(vx: HvxVector, vuu: HvxVectorPair, rt: i32) -> HvxVector {
    vdmpyhsuisat_acc(vx, vuu, rt)
}

/// `Vd32.w=vdmpy(Vu32.h,Rt32.uh):sat`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vdmpy_vhruh_sat(vu: HvxVector, rt: i32) -> HvxVector {
    vdmpyhsusat(vu, rt)
}

/// `Vx32.w+=vdmpy(Vu32.h,Rt32.uh):sat`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vdmpyacc_vwvhruh_sat(vx: HvxVector, vu: HvxVector, rt: i32) -> HvxVector {
    vdmpyhsusat_acc(vx, vu, rt)
}

/// `Vd32.w=vdmpy(Vu32.h,Vv32.h):sat`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vdmpy_vhvh_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vdmpyhvsat(vu, vv)
}

/// `Vx32.w+=vdmpy(Vu32.h,Vv32.h):sat`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vdmpyacc_vwvhvh_sat(vx: HvxVector, vu: HvxVector, vv: HvxVector) -> HvxVector {
    vdmpyhvsat_acc(vx, vu, vv)
}

/// `Vdd32.uw=vdsad(Vuu32.uh,Rt32.uh)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wuw_vdsad_wuhruh(vuu: HvxVectorPair, rt: i32) -> HvxVectorPair {
    vdsaduh(vuu, rt)
}

/// `Vxx32.uw+=vdsad(Vuu32.uh,Rt32.uh)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
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
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vinsert_vwr(vx: HvxVector, rt: i32) -> HvxVector {
    vinsertwr(vx, rt)
}

/// `Vd32=vlalign(Vu32,Vv32,Rt8)`
///
/// Instruction Type: CVI_VP
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_v_vlalign_vvr(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVector {
    vlalignb(vu, vv, rt)
}

/// `Vd32=vlalign(Vu32,Vv32,#u3)`
///
/// Instruction Type: CVI_VP
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_v_vlalign_vvi(vu: HvxVector, vv: HvxVector, iu3: i32) -> HvxVector {
    vlalignbi(vu, vv, iu3)
}

/// `Vd32.uh=vlsr(Vu32.uh,Rt32)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vuh_vlsr_vuhr(vu: HvxVector, rt: i32) -> HvxVector {
    vlsrh(vu, rt)
}

/// `Vd32.h=vlsr(Vu32.h,Vv32.h)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_vlsr_vhvh(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vlsrhv(vu, vv)
}

/// `Vd32.uw=vlsr(Vu32.uw,Rt32)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vuw_vlsr_vuwr(vu: HvxVector, rt: i32) -> HvxVector {
    vlsrw(vu, rt)
}

/// `Vd32.w=vlsr(Vu32.w,Vv32.w)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vlsr_vwvw(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vlsrwv(vu, vv)
}

/// `Vd32.b=vlut32(Vu32.b,Vv32.b,Rt8)`
///
/// Instruction Type: CVI_VP
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vb_vlut32_vbvbr(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVector {
    vlutvvb(vu, vv, rt)
}

/// `Vx32.b|=vlut32(Vu32.b,Vv32.b,Rt8)`
///
/// Instruction Type: CVI_VP_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
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
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wh_vlut16_vbvhr(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVectorPair {
    vlutvwh(vu, vv, rt)
}

/// `Vxx32.h|=vlut16(Vu32.b,Vv32.h,Rt8)`
///
/// Instruction Type: CVI_VP_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
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
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_vmax_vhvh(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmaxh(vu, vv)
}

/// `Vd32.ub=vmax(Vu32.ub,Vv32.ub)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vub_vmax_vubvub(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmaxub(vu, vv)
}

/// `Vd32.uh=vmax(Vu32.uh,Vv32.uh)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vuh_vmax_vuhvuh(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmaxuh(vu, vv)
}

/// `Vd32.w=vmax(Vu32.w,Vv32.w)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vmax_vwvw(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmaxw(vu, vv)
}

/// `Vd32.h=vmin(Vu32.h,Vv32.h)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_vmin_vhvh(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vminh(vu, vv)
}

/// `Vd32.ub=vmin(Vu32.ub,Vv32.ub)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vub_vmin_vubvub(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vminub(vu, vv)
}

/// `Vd32.uh=vmin(Vu32.uh,Vv32.uh)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vuh_vmin_vuhvuh(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vminuh(vu, vv)
}

/// `Vd32.w=vmin(Vu32.w,Vv32.w)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vmin_vwvw(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vminw(vu, vv)
}

/// `Vdd32.h=vmpa(Vuu32.ub,Rt32.b)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wh_vmpa_wubrb(vuu: HvxVectorPair, rt: i32) -> HvxVectorPair {
    vmpabus(vuu, rt)
}

/// `Vxx32.h+=vmpa(Vuu32.ub,Rt32.b)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
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
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wh_vmpa_wubwb(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
    vmpabusv(vuu, vvv)
}

/// `Vdd32.h=vmpa(Vuu32.ub,Vvv32.ub)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wh_vmpa_wubwub(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
    vmpabuuv(vuu, vvv)
}

/// `Vdd32.w=vmpa(Vuu32.h,Rt32.b)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_ww_vmpa_whrb(vuu: HvxVectorPair, rt: i32) -> HvxVectorPair {
    vmpahb(vuu, rt)
}

/// `Vxx32.w+=vmpa(Vuu32.h,Rt32.b)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
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
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wh_vmpy_vubrb(vu: HvxVector, rt: i32) -> HvxVectorPair {
    vmpybus(vu, rt)
}

/// `Vxx32.h+=vmpy(Vu32.ub,Rt32.b)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wh_vmpyacc_whvubrb(vxx: HvxVectorPair, vu: HvxVector, rt: i32) -> HvxVectorPair {
    vmpybus_acc(vxx, vu, rt)
}

/// `Vdd32.h=vmpy(Vu32.ub,Vv32.b)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wh_vmpy_vubvb(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
    vmpybusv(vu, vv)
}

/// `Vxx32.h+=vmpy(Vu32.ub,Vv32.b)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
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
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wh_vmpy_vbvb(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
    vmpybv(vu, vv)
}

/// `Vxx32.h+=vmpy(Vu32.b,Vv32.b)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
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
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vmpye_vwvuh(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmpyewuh(vu, vv)
}

/// `Vdd32.w=vmpy(Vu32.h,Rt32.h)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_ww_vmpy_vhrh(vu: HvxVector, rt: i32) -> HvxVectorPair {
    vmpyh(vu, rt)
}

/// `Vxx32.w+=vmpy(Vu32.h,Rt32.h):sat`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
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
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_vmpy_vhrh_s1_rnd_sat(vu: HvxVector, rt: i32) -> HvxVector {
    vmpyhsrs(vu, rt)
}

/// `Vd32.h=vmpy(Vu32.h,Rt32.h):<<1:sat`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_vmpy_vhrh_s1_sat(vu: HvxVector, rt: i32) -> HvxVector {
    vmpyhss(vu, rt)
}

/// `Vdd32.w=vmpy(Vu32.h,Vv32.uh)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_ww_vmpy_vhvuh(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
    vmpyhus(vu, vv)
}

/// `Vxx32.w+=vmpy(Vu32.h,Vv32.uh)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
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
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_ww_vmpy_vhvh(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
    vmpyhv(vu, vv)
}

/// `Vxx32.w+=vmpy(Vu32.h,Vv32.h)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
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
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_vmpy_vhvh_s1_rnd_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmpyhvsrs(vu, vv)
}

/// `Vd32.w=vmpyieo(Vu32.h,Vv32.h)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vmpyieo_vhvh(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmpyieoh(vu, vv)
}

/// `Vx32.w+=vmpyie(Vu32.w,Vv32.h)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vmpyieacc_vwvwvh(vx: HvxVector, vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmpyiewh_acc(vx, vu, vv)
}

/// `Vd32.w=vmpyie(Vu32.w,Vv32.uh)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vmpyie_vwvuh(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmpyiewuh(vu, vv)
}

/// `Vx32.w+=vmpyie(Vu32.w,Vv32.uh)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vmpyieacc_vwvwvuh(vx: HvxVector, vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmpyiewuh_acc(vx, vu, vv)
}

/// `Vd32.h=vmpyi(Vu32.h,Vv32.h)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_vmpyi_vhvh(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmpyih(vu, vv)
}

/// `Vx32.h+=vmpyi(Vu32.h,Vv32.h)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_vmpyiacc_vhvhvh(vx: HvxVector, vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmpyih_acc(vx, vu, vv)
}

/// `Vd32.h=vmpyi(Vu32.h,Rt32.b)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_vmpyi_vhrb(vu: HvxVector, rt: i32) -> HvxVector {
    vmpyihb(vu, rt)
}

/// `Vx32.h+=vmpyi(Vu32.h,Rt32.b)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_vmpyiacc_vhvhrb(vx: HvxVector, vu: HvxVector, rt: i32) -> HvxVector {
    vmpyihb_acc(vx, vu, rt)
}

/// `Vd32.w=vmpyio(Vu32.w,Vv32.h)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vmpyio_vwvh(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmpyiowh(vu, vv)
}

/// `Vd32.w=vmpyi(Vu32.w,Rt32.b)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vmpyi_vwrb(vu: HvxVector, rt: i32) -> HvxVector {
    vmpyiwb(vu, rt)
}

/// `Vx32.w+=vmpyi(Vu32.w,Rt32.b)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vmpyiacc_vwvwrb(vx: HvxVector, vu: HvxVector, rt: i32) -> HvxVector {
    vmpyiwb_acc(vx, vu, rt)
}

/// `Vd32.w=vmpyi(Vu32.w,Rt32.h)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vmpyi_vwrh(vu: HvxVector, rt: i32) -> HvxVector {
    vmpyiwh(vu, rt)
}

/// `Vx32.w+=vmpyi(Vu32.w,Rt32.h)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vmpyiacc_vwvwrh(vx: HvxVector, vu: HvxVector, rt: i32) -> HvxVector {
    vmpyiwh_acc(vx, vu, rt)
}

/// `Vd32.w=vmpyo(Vu32.w,Vv32.h):<<1:sat`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vmpyo_vwvh_s1_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmpyowh(vu, vv)
}

/// `Vd32.w=vmpyo(Vu32.w,Vv32.h):<<1:rnd:sat`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vmpyo_vwvh_s1_rnd_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmpyowh_rnd(vu, vv)
}

/// `Vx32.w+=vmpyo(Vu32.w,Vv32.h):<<1:rnd:sat:shift`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
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
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
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
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wuh_vmpy_vubrub(vu: HvxVector, rt: i32) -> HvxVectorPair {
    vmpyub(vu, rt)
}

/// `Vxx32.uh+=vmpy(Vu32.ub,Rt32.ub)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
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
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wuh_vmpy_vubvub(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
    vmpyubv(vu, vv)
}

/// `Vxx32.uh+=vmpy(Vu32.ub,Vv32.ub)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
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
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wuw_vmpy_vuhruh(vu: HvxVector, rt: i32) -> HvxVectorPair {
    vmpyuh(vu, rt)
}

/// `Vxx32.uw+=vmpy(Vu32.uh,Rt32.uh)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
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
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wuw_vmpy_vuhvuh(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
    vmpyuhv(vu, vv)
}

/// `Vxx32.uw+=vmpy(Vu32.uh,Vv32.uh)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
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
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_vnavg_vhvh(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vnavgh(vu, vv)
}

/// `Vd32.b=vnavg(Vu32.ub,Vv32.ub)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vb_vnavg_vubvub(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vnavgub(vu, vv)
}

/// `Vd32.w=vnavg(Vu32.w,Vv32.w)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vnavg_vwvw(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vnavgw(vu, vv)
}

/// `Vd32.h=vnormamt(Vu32.h)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_vnormamt_vh(vu: HvxVector) -> HvxVector {
    vnormamth(vu)
}

/// `Vd32.w=vnormamt(Vu32.w)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vnormamt_vw(vu: HvxVector) -> HvxVector {
    vnormamtw(vu)
}

/// `Vd32=vnot(Vu32)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_v_vnot_v(vu: HvxVector) -> HvxVector {
    vnot(vu)
}

/// `Vd32=vor(Vu32,Vv32)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_v_vor_vv(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vor(vu, vv)
}

/// `Vd32.b=vpacke(Vu32.h,Vv32.h)`
///
/// Instruction Type: CVI_VP
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vb_vpacke_vhvh(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vpackeb(vu, vv)
}

/// `Vd32.h=vpacke(Vu32.w,Vv32.w)`
///
/// Instruction Type: CVI_VP
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_vpacke_vwvw(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vpackeh(vu, vv)
}

/// `Vd32.b=vpack(Vu32.h,Vv32.h):sat`
///
/// Instruction Type: CVI_VP
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vb_vpack_vhvh_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vpackhb_sat(vu, vv)
}

/// `Vd32.ub=vpack(Vu32.h,Vv32.h):sat`
///
/// Instruction Type: CVI_VP
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vub_vpack_vhvh_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vpackhub_sat(vu, vv)
}

/// `Vd32.b=vpacko(Vu32.h,Vv32.h)`
///
/// Instruction Type: CVI_VP
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vb_vpacko_vhvh(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vpackob(vu, vv)
}

/// `Vd32.h=vpacko(Vu32.w,Vv32.w)`
///
/// Instruction Type: CVI_VP
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_vpacko_vwvw(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vpackoh(vu, vv)
}

/// `Vd32.h=vpack(Vu32.w,Vv32.w):sat`
///
/// Instruction Type: CVI_VP
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_vpack_vwvw_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vpackwh_sat(vu, vv)
}

/// `Vd32.uh=vpack(Vu32.w,Vv32.w):sat`
///
/// Instruction Type: CVI_VP
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vuh_vpack_vwvw_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vpackwuh_sat(vu, vv)
}

/// `Vd32.h=vpopcount(Vu32.h)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_vpopcount_vh(vu: HvxVector) -> HvxVector {
    vpopcounth(vu)
}

/// `Vd32=vrdelta(Vu32,Vv32)`
///
/// Instruction Type: CVI_VP
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_v_vrdelta_vv(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vrdelta(vu, vv)
}

/// `Vd32.w=vrmpy(Vu32.ub,Rt32.b)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vrmpy_vubrb(vu: HvxVector, rt: i32) -> HvxVector {
    vrmpybus(vu, rt)
}

/// `Vx32.w+=vrmpy(Vu32.ub,Rt32.b)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vrmpyacc_vwvubrb(vx: HvxVector, vu: HvxVector, rt: i32) -> HvxVector {
    vrmpybus_acc(vx, vu, rt)
}

/// `Vdd32.w=vrmpy(Vuu32.ub,Rt32.b,#u1)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_ww_vrmpy_wubrbi(vuu: HvxVectorPair, rt: i32, iu1: i32) -> HvxVectorPair {
    vrmpybusi(vuu, rt, iu1)
}

/// `Vxx32.w+=vrmpy(Vuu32.ub,Rt32.b,#u1)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
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
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vrmpy_vubvb(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vrmpybusv(vu, vv)
}

/// `Vx32.w+=vrmpy(Vu32.ub,Vv32.b)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vrmpyacc_vwvubvb(vx: HvxVector, vu: HvxVector, vv: HvxVector) -> HvxVector {
    vrmpybusv_acc(vx, vu, vv)
}

/// `Vd32.w=vrmpy(Vu32.b,Vv32.b)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vrmpy_vbvb(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vrmpybv(vu, vv)
}

/// `Vx32.w+=vrmpy(Vu32.b,Vv32.b)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vrmpyacc_vwvbvb(vx: HvxVector, vu: HvxVector, vv: HvxVector) -> HvxVector {
    vrmpybv_acc(vx, vu, vv)
}

/// `Vd32.uw=vrmpy(Vu32.ub,Rt32.ub)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vuw_vrmpy_vubrub(vu: HvxVector, rt: i32) -> HvxVector {
    vrmpyub(vu, rt)
}

/// `Vx32.uw+=vrmpy(Vu32.ub,Rt32.ub)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vuw_vrmpyacc_vuwvubrub(vx: HvxVector, vu: HvxVector, rt: i32) -> HvxVector {
    vrmpyub_acc(vx, vu, rt)
}

/// `Vdd32.uw=vrmpy(Vuu32.ub,Rt32.ub,#u1)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wuw_vrmpy_wubrubi(vuu: HvxVectorPair, rt: i32, iu1: i32) -> HvxVectorPair {
    vrmpyubi(vuu, rt, iu1)
}

/// `Vxx32.uw+=vrmpy(Vuu32.ub,Rt32.ub,#u1)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
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
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vuw_vrmpy_vubvub(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vrmpyubv(vu, vv)
}

/// `Vx32.uw+=vrmpy(Vu32.ub,Vv32.ub)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vuw_vrmpyacc_vuwvubvub(vx: HvxVector, vu: HvxVector, vv: HvxVector) -> HvxVector {
    vrmpyubv_acc(vx, vu, vv)
}

/// `Vd32=vror(Vu32,Rt32)`
///
/// Instruction Type: CVI_VP
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_v_vror_vr(vu: HvxVector, rt: i32) -> HvxVector {
    vror(vu, rt)
}

/// `Vd32.b=vround(Vu32.h,Vv32.h):sat`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vb_vround_vhvh_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vroundhb(vu, vv)
}

/// `Vd32.ub=vround(Vu32.h,Vv32.h):sat`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vub_vround_vhvh_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vroundhub(vu, vv)
}

/// `Vd32.h=vround(Vu32.w,Vv32.w):sat`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_vround_vwvw_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vroundwh(vu, vv)
}

/// `Vd32.uh=vround(Vu32.w,Vv32.w):sat`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vuh_vround_vwvw_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vroundwuh(vu, vv)
}

/// `Vdd32.uw=vrsad(Vuu32.ub,Rt32.ub,#u1)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wuw_vrsad_wubrubi(vuu: HvxVectorPair, rt: i32, iu1: i32) -> HvxVectorPair {
    vrsadubi(vuu, rt, iu1)
}

/// `Vxx32.uw+=vrsad(Vuu32.ub,Rt32.ub,#u1)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
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
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vub_vsat_vhvh(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vsathub(vu, vv)
}

/// `Vd32.h=vsat(Vu32.w,Vv32.w)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_vsat_vwvw(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vsatwh(vu, vv)
}

/// `Vdd32.h=vsxt(Vu32.b)`
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wh_vsxt_vb(vu: HvxVector) -> HvxVectorPair {
    vsb(vu)
}

/// `Vdd32.w=vsxt(Vu32.h)`
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_ww_vsxt_vh(vu: HvxVector) -> HvxVectorPair {
    vsh(vu)
}

/// `Vd32.h=vshuffe(Vu32.h,Vv32.h)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_vshuffe_vhvh(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vshufeh(vu, vv)
}

/// `Vd32.b=vshuff(Vu32.b)`
///
/// Instruction Type: CVI_VP
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vb_vshuff_vb(vu: HvxVector) -> HvxVector {
    vshuffb(vu)
}

/// `Vd32.b=vshuffe(Vu32.b,Vv32.b)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vb_vshuffe_vbvb(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vshuffeb(vu, vv)
}

/// `Vd32.h=vshuff(Vu32.h)`
///
/// Instruction Type: CVI_VP
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_vshuff_vh(vu: HvxVector) -> HvxVector {
    vshuffh(vu)
}

/// `Vd32.b=vshuffo(Vu32.b,Vv32.b)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vb_vshuffo_vbvb(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vshuffob(vu, vv)
}

/// `Vdd32=vshuff(Vu32,Vv32,Rt8)`
///
/// Instruction Type: CVI_VP_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_w_vshuff_vvr(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVectorPair {
    vshuffvdd(vu, vv, rt)
}

/// `Vdd32.b=vshuffoe(Vu32.b,Vv32.b)`
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wb_vshuffoe_vbvb(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
    vshufoeb(vu, vv)
}

/// `Vdd32.h=vshuffoe(Vu32.h,Vv32.h)`
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wh_vshuffoe_vhvh(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
    vshufoeh(vu, vv)
}

/// `Vd32.h=vshuffo(Vu32.h,Vv32.h)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_vshuffo_vhvh(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vshufoh(vu, vv)
}

/// `Vd32.b=vsub(Vu32.b,Vv32.b)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vb_vsub_vbvb(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vsubb(vu, vv)
}

/// `Vdd32.b=vsub(Vuu32.b,Vvv32.b)`
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wb_vsub_wbwb(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
    vsubb_dv(vuu, vvv)
}

/// `Vd32.h=vsub(Vu32.h,Vv32.h)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_vsub_vhvh(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vsubh(vu, vv)
}

/// `Vdd32.h=vsub(Vuu32.h,Vvv32.h)`
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wh_vsub_whwh(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
    vsubh_dv(vuu, vvv)
}

/// `Vd32.h=vsub(Vu32.h,Vv32.h):sat`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_vsub_vhvh_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vsubhsat(vu, vv)
}

/// `Vdd32.h=vsub(Vuu32.h,Vvv32.h):sat`
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wh_vsub_whwh_sat(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
    vsubhsat_dv(vuu, vvv)
}

/// `Vdd32.w=vsub(Vu32.h,Vv32.h)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_ww_vsub_vhvh(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
    vsubhw(vu, vv)
}

/// `Vdd32.h=vsub(Vu32.ub,Vv32.ub)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wh_vsub_vubvub(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
    vsububh(vu, vv)
}

/// `Vd32.ub=vsub(Vu32.ub,Vv32.ub):sat`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vub_vsub_vubvub_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vsububsat(vu, vv)
}

/// `Vdd32.ub=vsub(Vuu32.ub,Vvv32.ub):sat`
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wub_vsub_wubwub_sat(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
    vsububsat_dv(vuu, vvv)
}

/// `Vd32.uh=vsub(Vu32.uh,Vv32.uh):sat`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vuh_vsub_vuhvuh_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vsubuhsat(vu, vv)
}

/// `Vdd32.uh=vsub(Vuu32.uh,Vvv32.uh):sat`
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wuh_vsub_wuhwuh_sat(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
    vsubuhsat_dv(vuu, vvv)
}

/// `Vdd32.w=vsub(Vu32.uh,Vv32.uh)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_ww_vsub_vuhvuh(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
    vsubuhw(vu, vv)
}

/// `Vd32.w=vsub(Vu32.w,Vv32.w)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vsub_vwvw(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vsubw(vu, vv)
}

/// `Vdd32.w=vsub(Vuu32.w,Vvv32.w)`
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_ww_vsub_wwww(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
    vsubw_dv(vuu, vvv)
}

/// `Vd32.w=vsub(Vu32.w,Vv32.w):sat`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vsub_vwvw_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vsubwsat(vu, vv)
}

/// `Vdd32.w=vsub(Vuu32.w,Vvv32.w):sat`
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_ww_vsub_wwww_sat(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
    vsubwsat_dv(vuu, vvv)
}

/// `Vdd32.h=vtmpy(Vuu32.b,Rt32.b)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wh_vtmpy_wbrb(vuu: HvxVectorPair, rt: i32) -> HvxVectorPair {
    vtmpyb(vuu, rt)
}

/// `Vxx32.h+=vtmpy(Vuu32.b,Rt32.b)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
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
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wh_vtmpy_wubrb(vuu: HvxVectorPair, rt: i32) -> HvxVectorPair {
    vtmpybus(vuu, rt)
}

/// `Vxx32.h+=vtmpy(Vuu32.ub,Rt32.b)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
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
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_ww_vtmpy_whrb(vuu: HvxVectorPair, rt: i32) -> HvxVectorPair {
    vtmpyhb(vuu, rt)
}

/// `Vxx32.w+=vtmpy(Vuu32.h,Rt32.b)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
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
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wh_vunpack_vb(vu: HvxVector) -> HvxVectorPair {
    vunpackb(vu)
}

/// `Vdd32.w=vunpack(Vu32.h)`
///
/// Instruction Type: CVI_VP_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_ww_vunpack_vh(vu: HvxVector) -> HvxVectorPair {
    vunpackh(vu)
}

/// `Vxx32.h|=vunpacko(Vu32.b)`
///
/// Instruction Type: CVI_VP_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wh_vunpackoor_whvb(vxx: HvxVectorPair, vu: HvxVector) -> HvxVectorPair {
    vunpackob(vxx, vu)
}

/// `Vxx32.w|=vunpacko(Vu32.h)`
///
/// Instruction Type: CVI_VP_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_ww_vunpackoor_wwvh(vxx: HvxVectorPair, vu: HvxVector) -> HvxVectorPair {
    vunpackoh(vxx, vu)
}

/// `Vdd32.uh=vunpack(Vu32.ub)`
///
/// Instruction Type: CVI_VP_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wuh_vunpack_vub(vu: HvxVector) -> HvxVectorPair {
    vunpackub(vu)
}

/// `Vdd32.uw=vunpack(Vu32.uh)`
///
/// Instruction Type: CVI_VP_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wuw_vunpack_vuh(vu: HvxVector) -> HvxVectorPair {
    vunpackuh(vu)
}

/// `Vd32=vxor(Vu32,Vv32)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_v_vxor_vv(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vxor(vu, vv)
}

/// `Vdd32.uh=vzxt(Vu32.ub)`
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wuh_vzxt_vub(vu: HvxVector) -> HvxVectorPair {
    vzb(vu)
}

/// `Vdd32.uw=vzxt(Vu32.uh)`
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wuw_vzxt_vuh(vu: HvxVector) -> HvxVectorPair {
    vzh(vu)
}

/// `Qd4=and(Qs4,Qt4)` (compound operation)
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_and_qq(qs: HvxVectorPred, qt: HvxVectorPred) -> HvxVectorPred {
    pred_and(qs, qt)
}

/// `Qd4=and(Qs4,!Qt4)` (compound operation)
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_and_qqn(qs: HvxVectorPred, qt: HvxVectorPred) -> HvxVectorPred {
    pred_and_n(qs, qt)
}

/// `Qd4=not(Qs4)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_not_q(qs: HvxVectorPred) -> HvxVectorPred {
    pred_not(qs)
}

/// `Qd4=or(Qs4,Qt4)` (compound operation)
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_or_qq(qs: HvxVectorPred, qt: HvxVectorPred) -> HvxVectorPred {
    pred_or(qs, qt)
}

/// `Qd4=or(Qs4,!Qt4)` (compound operation)
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_or_qqn(qs: HvxVectorPred, qt: HvxVectorPred) -> HvxVectorPred {
    pred_or_n(qs, qt)
}

/// `Qd4=vsetq(Rt32)` (compound operation)
///
/// Instruction Type: CVI_VP
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vsetq_r(rt: i32) -> HvxVectorPred {
    pred_scalar2(rt)
}

/// `Qd4=xor(Qs4,Qt4)` (compound operation)
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_xor_qq(qs: HvxVectorPred, qt: HvxVectorPred) -> HvxVectorPred {
    pred_xor(qs, qt)
}

/// `if (!Qv4) vmem(Rt32+#s4)=Vs32` (compound operation)
///
/// Instruction Type: CVI_VM_ST
/// Execution Slots: SLOT0
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vmem_qnriv(qv: HvxVectorPred, rt: *mut HvxVector, vs: HvxVector) -> () {
    vS32b_nqpred_ai(qv, rt, vs)
}

/// `if (!Qv4) vmem(Rt32+#s4):nt=Vs32` (compound operation)
///
/// Instruction Type: CVI_VM_ST
/// Execution Slots: SLOT0
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vmem_qnriv_nt(qv: HvxVectorPred, rt: *mut HvxVector, vs: HvxVector) -> () {
    vS32b_nt_nqpred_ai(qv, rt, vs)
}

/// `if (Qv4) vmem(Rt32+#s4):nt=Vs32` (compound operation)
///
/// Instruction Type: CVI_VM_ST
/// Execution Slots: SLOT0
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vmem_qriv_nt(qv: HvxVectorPred, rt: *mut HvxVector, vs: HvxVector) -> () {
    vS32b_nt_qpred_ai(qv, rt, vs)
}

/// `if (Qv4) vmem(Rt32+#s4)=Vs32` (compound operation)
///
/// Instruction Type: CVI_VM_ST
/// Execution Slots: SLOT0
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vmem_qriv(qv: HvxVectorPred, rt: *mut HvxVector, vs: HvxVector) -> () {
    vS32b_qpred_ai(qv, rt, vs)
}

/// `if (!Qv4) Vx32.b+=Vu32.b` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vb_condacc_qnvbvb(qv: HvxVectorPred, vx: HvxVector, vu: HvxVector) -> HvxVector {
    vaddbnq(qv, vx, vu)
}

/// `if (Qv4) Vx32.b+=Vu32.b` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vb_condacc_qvbvb(qv: HvxVectorPred, vx: HvxVector, vu: HvxVector) -> HvxVector {
    vaddbq(qv, vx, vu)
}

/// `if (!Qv4) Vx32.h+=Vu32.h` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_condacc_qnvhvh(qv: HvxVectorPred, vx: HvxVector, vu: HvxVector) -> HvxVector {
    vaddhnq(qv, vx, vu)
}

/// `if (Qv4) Vx32.h+=Vu32.h` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_condacc_qvhvh(qv: HvxVectorPred, vx: HvxVector, vu: HvxVector) -> HvxVector {
    vaddhq(qv, vx, vu)
}

/// `if (!Qv4) Vx32.w+=Vu32.w` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_condacc_qnvwvw(qv: HvxVectorPred, vx: HvxVector, vu: HvxVector) -> HvxVector {
    vaddwnq(qv, vx, vu)
}

/// `if (Qv4) Vx32.w+=Vu32.w` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_condacc_qvwvw(qv: HvxVectorPred, vx: HvxVector, vu: HvxVector) -> HvxVector {
    vaddwq(qv, vx, vu)
}

/// `Vd32=vand(Qu4,Rt32)` (compound operation)
///
/// Instruction Type: CVI_VX_LATE
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_v_vand_qr(qu: HvxVectorPred, rt: i32) -> HvxVector {
    vandqrt(qu, rt)
}

/// `Vx32|=vand(Qu4,Rt32)` (compound operation)
///
/// Instruction Type: CVI_VX_LATE
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_v_vandor_vqr(vx: HvxVector, qu: HvxVectorPred, rt: i32) -> HvxVector {
    vandqrt_acc(vx, qu, rt)
}

/// `Qd4=vand(Vu32,Rt32)` (compound operation)
///
/// Instruction Type: CVI_VX_LATE
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vand_vr(vu: HvxVector, rt: i32) -> HvxVectorPred {
    vandqrt(vu, rt)
}

/// `Qx4|=vand(Vu32,Rt32)` (compound operation)
///
/// Instruction Type: CVI_VX_LATE
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vandor_qvr(qx: HvxVectorPred, vu: HvxVector, rt: i32) -> HvxVectorPred {
    vandqrt(qx, vu, rt)
}

/// `Qd4=vcmp.eq(Vu32.b,Vv32.b)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_eq_vbvb(vu: HvxVector, vv: HvxVector) -> HvxVectorPred {
    veqb(vu, vv)
}

/// `Qx4&=vcmp.eq(Vu32.b,Vv32.b)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_eqand_qvbvb(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    veqb_and(qx, vu, vv)
}

/// `Qx4|=vcmp.eq(Vu32.b,Vv32.b)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_eqor_qvbvb(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    veqb_or(qx, vu, vv)
}

/// `Qx4^=vcmp.eq(Vu32.b,Vv32.b)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_eqxacc_qvbvb(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    veqb_xor(qx, vu, vv)
}

/// `Qd4=vcmp.eq(Vu32.h,Vv32.h)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_eq_vhvh(vu: HvxVector, vv: HvxVector) -> HvxVectorPred {
    veqh(vu, vv)
}

/// `Qx4&=vcmp.eq(Vu32.h,Vv32.h)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_eqand_qvhvh(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    veqh_and(qx, vu, vv)
}

/// `Qx4|=vcmp.eq(Vu32.h,Vv32.h)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_eqor_qvhvh(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    veqh_or(qx, vu, vv)
}

/// `Qx4^=vcmp.eq(Vu32.h,Vv32.h)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_eqxacc_qvhvh(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    veqh_xor(qx, vu, vv)
}

/// `Qd4=vcmp.eq(Vu32.w,Vv32.w)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_eq_vwvw(vu: HvxVector, vv: HvxVector) -> HvxVectorPred {
    veqw(vu, vv)
}

/// `Qx4&=vcmp.eq(Vu32.w,Vv32.w)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_eqand_qvwvw(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    veqw_and(qx, vu, vv)
}

/// `Qx4|=vcmp.eq(Vu32.w,Vv32.w)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_eqor_qvwvw(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    veqw_or(qx, vu, vv)
}

/// `Qx4^=vcmp.eq(Vu32.w,Vv32.w)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_eqxacc_qvwvw(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    veqw_xor(qx, vu, vv)
}

/// `Qd4=vcmp.gt(Vu32.b,Vv32.b)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_gt_vbvb(vu: HvxVector, vv: HvxVector) -> HvxVectorPred {
    vgtb(vu, vv)
}

/// `Qx4&=vcmp.gt(Vu32.b,Vv32.b)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_gtand_qvbvb(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    vgtb_and(qx, vu, vv)
}

/// `Qx4|=vcmp.gt(Vu32.b,Vv32.b)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_gtor_qvbvb(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    vgtb_or(qx, vu, vv)
}

/// `Qx4^=vcmp.gt(Vu32.b,Vv32.b)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_gtxacc_qvbvb(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    vgtb_xor(qx, vu, vv)
}

/// `Qd4=vcmp.gt(Vu32.h,Vv32.h)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_gt_vhvh(vu: HvxVector, vv: HvxVector) -> HvxVectorPred {
    vgth(vu, vv)
}

/// `Qx4&=vcmp.gt(Vu32.h,Vv32.h)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_gtand_qvhvh(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    vgth_and(qx, vu, vv)
}

/// `Qx4|=vcmp.gt(Vu32.h,Vv32.h)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_gtor_qvhvh(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    vgth_or(qx, vu, vv)
}

/// `Qx4^=vcmp.gt(Vu32.h,Vv32.h)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_gtxacc_qvhvh(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    vgth_xor(qx, vu, vv)
}

/// `Qd4=vcmp.gt(Vu32.ub,Vv32.ub)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_gt_vubvub(vu: HvxVector, vv: HvxVector) -> HvxVectorPred {
    vgtub(vu, vv)
}

/// `Qx4&=vcmp.gt(Vu32.ub,Vv32.ub)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_gtand_qvubvub(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    vgtub_and(qx, vu, vv)
}

/// `Qx4|=vcmp.gt(Vu32.ub,Vv32.ub)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_gtor_qvubvub(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    vgtub_or(qx, vu, vv)
}

/// `Qx4^=vcmp.gt(Vu32.ub,Vv32.ub)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_gtxacc_qvubvub(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    vgtub_xor(qx, vu, vv)
}

/// `Qd4=vcmp.gt(Vu32.uh,Vv32.uh)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_gt_vuhvuh(vu: HvxVector, vv: HvxVector) -> HvxVectorPred {
    vgtuh(vu, vv)
}

/// `Qx4&=vcmp.gt(Vu32.uh,Vv32.uh)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_gtand_qvuhvuh(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    vgtuh_and(qx, vu, vv)
}

/// `Qx4|=vcmp.gt(Vu32.uh,Vv32.uh)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_gtor_qvuhvuh(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    vgtuh_or(qx, vu, vv)
}

/// `Qx4^=vcmp.gt(Vu32.uh,Vv32.uh)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_gtxacc_qvuhvuh(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    vgtuh_xor(qx, vu, vv)
}

/// `Qd4=vcmp.gt(Vu32.uw,Vv32.uw)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_gt_vuwvuw(vu: HvxVector, vv: HvxVector) -> HvxVectorPred {
    vgtuw(vu, vv)
}

/// `Qx4&=vcmp.gt(Vu32.uw,Vv32.uw)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_gtand_qvuwvuw(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    vgtuw_and(qx, vu, vv)
}

/// `Qx4|=vcmp.gt(Vu32.uw,Vv32.uw)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_gtor_qvuwvuw(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    vgtuw_or(qx, vu, vv)
}

/// `Qx4^=vcmp.gt(Vu32.uw,Vv32.uw)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_gtxacc_qvuwvuw(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    vgtuw_xor(qx, vu, vv)
}

/// `Qd4=vcmp.gt(Vu32.w,Vv32.w)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_gt_vwvw(vu: HvxVector, vv: HvxVector) -> HvxVectorPred {
    vgtw(vu, vv)
}

/// `Qx4&=vcmp.gt(Vu32.w,Vv32.w)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_gtand_qvwvw(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    vgtw_and(qx, vu, vv)
}

/// `Qx4|=vcmp.gt(Vu32.w,Vv32.w)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_gtor_qvwvw(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    vgtw_or(qx, vu, vv)
}

/// `Qx4^=vcmp.gt(Vu32.w,Vv32.w)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_gtxacc_qvwvw(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    vgtw_xor(qx, vu, vv)
}

/// `Vd32=vmux(Qt4,Vu32,Vv32)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_v_vmux_qvv(qt: HvxVectorPred, vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmux(qt, vu, vv)
}

/// `if (!Qv4) Vx32.b-=Vu32.b` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vb_condnac_qnvbvb(qv: HvxVectorPred, vx: HvxVector, vu: HvxVector) -> HvxVector {
    vsubbnq(qv, vx, vu)
}

/// `if (Qv4) Vx32.b-=Vu32.b` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vb_condnac_qvbvb(qv: HvxVectorPred, vx: HvxVector, vu: HvxVector) -> HvxVector {
    vsubbq(qv, vx, vu)
}

/// `if (!Qv4) Vx32.h-=Vu32.h` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_condnac_qnvhvh(qv: HvxVectorPred, vx: HvxVector, vu: HvxVector) -> HvxVector {
    vsubhnq(qv, vx, vu)
}

/// `if (Qv4) Vx32.h-=Vu32.h` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_condnac_qvhvh(qv: HvxVectorPred, vx: HvxVector, vu: HvxVector) -> HvxVector {
    vsubhq(qv, vx, vu)
}

/// `if (!Qv4) Vx32.w-=Vu32.w` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_condnac_qnvwvw(qv: HvxVectorPred, vx: HvxVector, vu: HvxVector) -> HvxVector {
    vsubwnq(qv, vx, vu)
}

/// `if (Qv4) Vx32.w-=Vu32.w` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_condnac_qvwvw(qv: HvxVectorPred, vx: HvxVector, vu: HvxVector) -> HvxVector {
    vsubwq(qv, vx, vu)
}

/// `Vdd32=vswap(Qt4,Vu32,Vv32)` (compound operation)
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv60"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_w_vswap_qvv(qt: HvxVectorPred, vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
    vswap(qt, vu, vv)
}

// ============================================================================
// HVX Architecture v62 intrinsics
// Target feature: hvxv62
// ============================================================================

/// `Vd32.b=vsplat(Rt32)`
///
/// Instruction Type: CVI_VX_LATE
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vb_vsplat_r(rt: i32) -> HvxVector {
    lvsplatb(rt)
}

/// `Vd32.h=vsplat(Rt32)`
///
/// Instruction Type: CVI_VX_LATE
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_vsplat_r(rt: i32) -> HvxVector {
    lvsplath(rt)
}

/// `Vd32.b=vadd(Vu32.b,Vv32.b):sat`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vb_vadd_vbvb_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vaddbsat(vu, vv)
}

/// `Vdd32.b=vadd(Vuu32.b,Vvv32.b):sat`
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wb_vadd_wbwb_sat(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
    vaddbsat_dv(vuu, vvv)
}

/// `Vd32.w=vadd(Vu32.w,Vv32.w,Qx4):carry`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vadd_vwvwq_carry(
    vu: HvxVector,
    vv: HvxVector,
    qx: *mut HvxVectorPred,
) -> HvxVector {
    vaddcarry(vu, vv, qx)
}

/// `Vd32.h=vadd(vclb(Vu32.h),Vv32.h)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_vadd_vclb_vhvh(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vaddclbh(vu, vv)
}

/// `Vd32.w=vadd(vclb(Vu32.w),Vv32.w)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vadd_vclb_vwvw(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vaddclbw(vu, vv)
}

/// `Vxx32.w+=vadd(Vu32.h,Vv32.h)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
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
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
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
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vub_vadd_vubvb_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vaddububb_sat(vu, vv)
}

/// `Vxx32.w+=vadd(Vu32.uh,Vv32.uh)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
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
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vuw_vadd_vuwvuw_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vadduwsat(vu, vv)
}

/// `Vdd32.uw=vadd(Vuu32.uw,Vvv32.uw):sat`
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wuw_vadd_wuwwuw_sat(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
    vadduwsat_dv(vuu, vvv)
}

/// `Vd32.b=vasr(Vu32.h,Vv32.h,Rt8):sat`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vb_vasr_vhvhr_sat(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVector {
    vasrhbsat(vu, vv, rt)
}

/// `Vd32.uh=vasr(Vu32.uw,Vv32.uw,Rt8):rnd:sat`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vuh_vasr_vuwvuwr_rnd_sat(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVector {
    vasruwuhrndsat(vu, vv, rt)
}

/// `Vd32.uh=vasr(Vu32.w,Vv32.w,Rt8):rnd:sat`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vuh_vasr_vwvwr_rnd_sat(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVector {
    vasrwuhrndsat(vu, vv, rt)
}

/// `Vd32.ub=vlsr(Vu32.ub,Rt32)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vub_vlsr_vubr(vu: HvxVector, rt: i32) -> HvxVector {
    vlsrb(vu, rt)
}

/// `Vd32.b=vlut32(Vu32.b,Vv32.b,Rt8):nomatch`
///
/// Instruction Type: CVI_VP
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vb_vlut32_vbvbr_nomatch(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVector {
    vlutvvb_nm(vu, vv, rt)
}

/// `Vx32.b|=vlut32(Vu32.b,Vv32.b,#u3)`
///
/// Instruction Type: CVI_VP_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
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
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vb_vlut32_vbvbi(vu: HvxVector, vv: HvxVector, iu3: i32) -> HvxVector {
    vlutvvbi(vu, vv, iu3)
}

/// `Vdd32.h=vlut16(Vu32.b,Vv32.h,Rt8):nomatch`
///
/// Instruction Type: CVI_VP_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wh_vlut16_vbvhr_nomatch(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVectorPair {
    vlutvwh_nm(vu, vv, rt)
}

/// `Vxx32.h|=vlut16(Vu32.b,Vv32.h,#u3)`
///
/// Instruction Type: CVI_VP_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
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
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wh_vlut16_vbvhi(vu: HvxVector, vv: HvxVector, iu3: i32) -> HvxVectorPair {
    vlutvwhi(vu, vv, iu3)
}

/// `Vd32.b=vmax(Vu32.b,Vv32.b)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vb_vmax_vbvb(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmaxb(vu, vv)
}

/// `Vd32.b=vmin(Vu32.b,Vv32.b)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vb_vmin_vbvb(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vminb(vu, vv)
}

/// `Vdd32.w=vmpa(Vuu32.uh,Rt32.b)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_ww_vmpa_wuhrb(vuu: HvxVectorPair, rt: i32) -> HvxVectorPair {
    vmpauhb(vuu, rt)
}

/// `Vxx32.w+=vmpa(Vuu32.uh,Rt32.b)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
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
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_w_vmpye_vwvuh(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
    vmpyewuh_64(vu, vv)
}

/// `Vd32.w=vmpyi(Vu32.w,Rt32.ub)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vmpyi_vwrub(vu: HvxVector, rt: i32) -> HvxVector {
    vmpyiwub(vu, rt)
}

/// `Vx32.w+=vmpyi(Vu32.w,Rt32.ub)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vmpyiacc_vwvwrub(vx: HvxVector, vu: HvxVector, rt: i32) -> HvxVector {
    vmpyiwub_acc(vx, vu, rt)
}

/// `Vxx32+=vmpyo(Vu32.w,Vv32.h)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
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
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vub_vround_vuhvuh_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vrounduhub(vu, vv)
}

/// `Vd32.uh=vround(Vu32.uw,Vv32.uw):sat`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vuh_vround_vuwvuw_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vrounduwuh(vu, vv)
}

/// `Vd32.uh=vsat(Vu32.uw,Vv32.uw)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vuh_vsat_vuwvuw(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vsatuwuh(vu, vv)
}

/// `Vd32.b=vsub(Vu32.b,Vv32.b):sat`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vb_vsub_vbvb_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vsubbsat(vu, vv)
}

/// `Vdd32.b=vsub(Vuu32.b,Vvv32.b):sat`
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wb_vsub_wbwb_sat(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
    vsubbsat_dv(vuu, vvv)
}

/// `Vd32.w=vsub(Vu32.w,Vv32.w,Qx4):carry`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vsub_vwvwq_carry(
    vu: HvxVector,
    vv: HvxVector,
    qx: *mut HvxVectorPred,
) -> HvxVector {
    vsubcarry(vu, vv, qx)
}

/// `Vd32.ub=vsub(Vu32.ub,Vv32.b):sat`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vub_vsub_vubvb_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vsubububb_sat(vu, vv)
}

/// `Vd32.uw=vsub(Vu32.uw,Vv32.uw):sat`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vuw_vsub_vuwvuw_sat(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vsubuwsat(vu, vv)
}

/// `Vdd32.uw=vsub(Vuu32.uw,Vvv32.uw):sat`
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wuw_vsub_wuwwuw_sat(vuu: HvxVectorPair, vvv: HvxVectorPair) -> HvxVectorPair {
    vsubuwsat_dv(vuu, vvv)
}

/// `Qd4=vsetq2(Rt32)` (compound operation)
///
/// Instruction Type: CVI_VP
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vsetq2_r(rt: i32) -> HvxVectorPred {
    pred_scalar2v2(rt)
}

/// `Qd4.b=vshuffe(Qs4.h,Qt4.h)` (compound operation)
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_qb_vshuffe_qhqh(qs: HvxVectorPred, qt: HvxVectorPred) -> HvxVectorPred {
    shuffeqh(qs, qt)
}

/// `Qd4.h=vshuffe(Qs4.w,Qt4.w)` (compound operation)
///
/// Instruction Type: CVI_VA_DV
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_qh_vshuffe_qwqw(qs: HvxVectorPred, qt: HvxVectorPred) -> HvxVectorPred {
    shuffeqw(qs, qt)
}

/// `Vd32=vand(!Qu4,Rt32)` (compound operation)
///
/// Instruction Type: CVI_VX_LATE
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_v_vand_qnr(qu: HvxVectorPred, rt: i32) -> HvxVector {
    vandnqrt(qu, rt)
}

/// `Vx32|=vand(!Qu4,Rt32)` (compound operation)
///
/// Instruction Type: CVI_VX_LATE
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_v_vandor_vqnr(vx: HvxVector, qu: HvxVectorPred, rt: i32) -> HvxVector {
    vandnqrt_acc(vx, qu, rt)
}

/// `Vd32=vand(!Qv4,Vu32)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_v_vand_qnv(qv: HvxVectorPred, vu: HvxVector) -> HvxVector {
    vandvnqv(qv, vu)
}

/// `Vd32=vand(Qv4,Vu32)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv62"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_v_vand_qv(qv: HvxVectorPred, vu: HvxVector) -> HvxVector {
    vandvqv(qv, vu)
}

// ============================================================================
// HVX Architecture v65 intrinsics
// Target feature: hvxv65
// ============================================================================

/// `Vd32.b=vabs(Vu32.b)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vb_vabs_vb(vu: HvxVector) -> HvxVector {
    vabsb(vu)
}

/// `Vd32.b=vabs(Vu32.b):sat`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vb_vabs_vb_sat(vu: HvxVector) -> HvxVector {
    vabsb_sat(vu)
}

/// `Vx32.h+=vasl(Vu32.h,Rt32)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_vaslacc_vhvhr(vx: HvxVector, vu: HvxVector, rt: i32) -> HvxVector {
    vaslh_acc(vx, vu, rt)
}

/// `Vx32.h+=vasr(Vu32.h,Rt32)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_vasracc_vhvhr(vx: HvxVector, vu: HvxVector, rt: i32) -> HvxVector {
    vasrh_acc(vx, vu, rt)
}

/// `Vd32.ub=vasr(Vu32.uh,Vv32.uh,Rt8):rnd:sat`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vub_vasr_vuhvuhr_rnd_sat(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVector {
    vasruhubrndsat(vu, vv, rt)
}

/// `Vd32.ub=vasr(Vu32.uh,Vv32.uh,Rt8):sat`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vub_vasr_vuhvuhr_sat(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVector {
    vasruhubsat(vu, vv, rt)
}

/// `Vd32.uh=vasr(Vu32.uw,Vv32.uw,Rt8):sat`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vuh_vasr_vuwvuwr_sat(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVector {
    vasruwuhsat(vu, vv, rt)
}

/// `Vd32.b=vavg(Vu32.b,Vv32.b)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vb_vavg_vbvb(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vavgb(vu, vv)
}

/// `Vd32.b=vavg(Vu32.b,Vv32.b):rnd`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vb_vavg_vbvb_rnd(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vavgbrnd(vu, vv)
}

/// `Vd32.uw=vavg(Vu32.uw,Vv32.uw)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vuw_vavg_vuwvuw(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vavguw(vu, vv)
}

/// `Vd32.uw=vavg(Vu32.uw,Vv32.uw):rnd`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vuw_vavg_vuwvuw_rnd(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vavguwrnd(vu, vv)
}

/// `Vdd32=#0`
///
/// Instruction Type: MAPPING
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_w_vzero() -> HvxVectorPair {
    vdd0()
}

/// `vtmp.h=vgather(Rt32,Mu2,Vv32.h).h`
///
/// Instruction Type: CVI_GATHER
/// Execution Slots: SLOT01
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vgather_armvh(rs: *mut HvxVector, rt: i32, mu: i32, vv: HvxVector) -> () {
    vgathermh(rs, rt, mu, vv)
}

/// `vtmp.h=vgather(Rt32,Mu2,Vvv32.w).h`
///
/// Instruction Type: CVI_GATHER_DV
/// Execution Slots: SLOT01
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vgather_armww(rs: *mut HvxVector, rt: i32, mu: i32, vvv: HvxVectorPair) -> () {
    vgathermhw(rs, rt, mu, vvv)
}

/// `vtmp.w=vgather(Rt32,Mu2,Vv32.w).w`
///
/// Instruction Type: CVI_GATHER
/// Execution Slots: SLOT01
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vgather_armvw(rs: *mut HvxVector, rt: i32, mu: i32, vv: HvxVector) -> () {
    vgathermw(rs, rt, mu, vv)
}

/// `Vd32.h=vlut4(Vu32.uh,Rtt32.h)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT2
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_vlut4_vuhph(vu: HvxVector, rtt: i64) -> HvxVector {
    vlut4(vu, rtt)
}

/// `Vdd32.h=vmpa(Vuu32.ub,Rt32.ub)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wh_vmpa_wubrub(vuu: HvxVectorPair, rt: i32) -> HvxVectorPair {
    vmpabuu(vuu, rt)
}

/// `Vxx32.h+=vmpa(Vuu32.ub,Rt32.ub)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wh_vmpaacc_whwubrub(
    vxx: HvxVectorPair,
    vuu: HvxVectorPair,
    rt: i32,
) -> HvxVectorPair {
    vmpabuu_acc(vxx, vuu, rt)
}

/// `Vx32.h=vmpa(Vx32.h,Vu32.h,Rtt32.h):sat`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT2
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_vmpa_vhvhvhph_sat(vx: HvxVector, vu: HvxVector, rtt: i64) -> HvxVector {
    vmpahhsat(vx, vu, rtt)
}

/// `Vx32.h=vmpa(Vx32.h,Vu32.uh,Rtt32.uh):sat`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT2
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_vmpa_vhvhvuhpuh_sat(vx: HvxVector, vu: HvxVector, rtt: i64) -> HvxVector {
    vmpauhuhsat(vx, vu, rtt)
}

/// `Vx32.h=vmps(Vx32.h,Vu32.uh,Rtt32.uh):sat`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT2
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_vmps_vhvhvuhpuh_sat(vx: HvxVector, vu: HvxVector, rtt: i64) -> HvxVector {
    vmpsuhuhsat(vx, vu, rtt)
}

/// `Vxx32.w+=vmpy(Vu32.h,Rt32.h)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_ww_vmpyacc_wwvhrh(vxx: HvxVectorPair, vu: HvxVector, rt: i32) -> HvxVectorPair {
    vmpyh_acc(vxx, vu, rt)
}

/// `Vd32.uw=vmpye(Vu32.uh,Rt32.uh)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vuw_vmpye_vuhruh(vu: HvxVector, rt: i32) -> HvxVector {
    vmpyuhe(vu, rt)
}

/// `Vx32.uw+=vmpye(Vu32.uh,Rt32.uh)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vuw_vmpyeacc_vuwvuhruh(vx: HvxVector, vu: HvxVector, rt: i32) -> HvxVector {
    vmpyuhe_acc(vx, vu, rt)
}

/// `Vd32.b=vnavg(Vu32.b,Vv32.b)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vb_vnavg_vbvb(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vnavgb(vu, vv)
}

/// `vscatter(Rt32,Mu2,Vv32.h).h=Vw32`
///
/// Instruction Type: CVI_SCATTER
/// Execution Slots: SLOT0
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vscatter_rmvhv(rt: i32, mu: i32, vv: HvxVector, vw: HvxVector) -> () {
    vscattermh(rt, mu, vv, vw)
}

/// `vscatter(Rt32,Mu2,Vv32.h).h+=Vw32`
///
/// Instruction Type: CVI_SCATTER
/// Execution Slots: SLOT0
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vscatteracc_rmvhv(rt: i32, mu: i32, vv: HvxVector, vw: HvxVector) -> () {
    vscattermh_add(rt, mu, vv, vw)
}

/// `vscatter(Rt32,Mu2,Vvv32.w).h=Vw32`
///
/// Instruction Type: CVI_SCATTER_DV
/// Execution Slots: SLOT0
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vscatter_rmwwv(rt: i32, mu: i32, vvv: HvxVectorPair, vw: HvxVector) -> () {
    vscattermhw(rt, mu, vvv, vw)
}

/// `vscatter(Rt32,Mu2,Vvv32.w).h+=Vw32`
///
/// Instruction Type: CVI_SCATTER_DV
/// Execution Slots: SLOT0
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vscatteracc_rmwwv(rt: i32, mu: i32, vvv: HvxVectorPair, vw: HvxVector) -> () {
    vscattermhw_add(rt, mu, vvv, vw)
}

/// `vscatter(Rt32,Mu2,Vv32.w).w=Vw32`
///
/// Instruction Type: CVI_SCATTER
/// Execution Slots: SLOT0
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vscatter_rmvwv(rt: i32, mu: i32, vv: HvxVector, vw: HvxVector) -> () {
    vscattermw(rt, mu, vv, vw)
}

/// `vscatter(Rt32,Mu2,Vv32.w).w+=Vw32`
///
/// Instruction Type: CVI_SCATTER
/// Execution Slots: SLOT0
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vscatteracc_rmvwv(rt: i32, mu: i32, vv: HvxVector, vw: HvxVector) -> () {
    vscattermw_add(rt, mu, vv, vw)
}

/// `if (Qs4) vtmp.h=vgather(Rt32,Mu2,Vv32.h).h` (compound operation)
///
/// Instruction Type: CVI_GATHER
/// Execution Slots: SLOT01
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vgather_aqrmvh(
    rs: *mut HvxVector,
    qs: HvxVectorPred,
    rt: i32,
    mu: i32,
    vv: HvxVector,
) -> () {
    vgathermhq(rs, qs, rt, mu, vv)
}

/// `if (Qs4) vtmp.h=vgather(Rt32,Mu2,Vvv32.w).h` (compound operation)
///
/// Instruction Type: CVI_GATHER_DV
/// Execution Slots: SLOT01
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vgather_aqrmww(
    rs: *mut HvxVector,
    qs: HvxVectorPred,
    rt: i32,
    mu: i32,
    vvv: HvxVectorPair,
) -> () {
    vgathermhwq(rs, qs, rt, mu, vvv)
}

/// `if (Qs4) vtmp.w=vgather(Rt32,Mu2,Vv32.w).w` (compound operation)
///
/// Instruction Type: CVI_GATHER
/// Execution Slots: SLOT01
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vgather_aqrmvw(
    rs: *mut HvxVector,
    qs: HvxVectorPred,
    rt: i32,
    mu: i32,
    vv: HvxVector,
) -> () {
    vgathermwq(rs, qs, rt, mu, vv)
}

/// `Vd32.b=prefixsum(Qv4)` (compound operation)
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vb_prefixsum_q(qv: HvxVectorPred) -> HvxVector {
    vprefixqb(qv)
}

/// `Vd32.h=prefixsum(Qv4)` (compound operation)
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_prefixsum_q(qv: HvxVectorPred) -> HvxVector {
    vprefixqh(qv)
}

/// `Vd32.w=prefixsum(Qv4)` (compound operation)
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_prefixsum_q(qv: HvxVectorPred) -> HvxVector {
    vprefixqw(qv)
}

/// `if (Qs4) vscatter(Rt32,Mu2,Vv32.h).h=Vw32` (compound operation)
///
/// Instruction Type: CVI_SCATTER
/// Execution Slots: SLOT0
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vscatter_qrmvhv(
    qs: HvxVectorPred,
    rt: i32,
    mu: i32,
    vv: HvxVector,
    vw: HvxVector,
) -> () {
    vscattermhq(qs, rt, mu, vv, vw)
}

/// `if (Qs4) vscatter(Rt32,Mu2,Vvv32.w).h=Vw32` (compound operation)
///
/// Instruction Type: CVI_SCATTER_DV
/// Execution Slots: SLOT0
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vscatter_qrmwwv(
    qs: HvxVectorPred,
    rt: i32,
    mu: i32,
    vvv: HvxVectorPair,
    vw: HvxVector,
) -> () {
    vscattermhwq(qs, rt, mu, vvv, vw)
}

/// `if (Qs4) vscatter(Rt32,Mu2,Vv32.w).w=Vw32` (compound operation)
///
/// Instruction Type: CVI_SCATTER
/// Execution Slots: SLOT0
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv65"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vscatter_qrmvwv(
    qs: HvxVectorPred,
    rt: i32,
    mu: i32,
    vv: HvxVector,
    vw: HvxVector,
) -> () {
    vscattermwq(qs, rt, mu, vv, vw)
}

// ============================================================================
// HVX Architecture v66 intrinsics
// Target feature: hvxv66
// ============================================================================

/// `Vxx32.w=vasrinto(Vu32.w,Vv32.w)`
///
/// Instruction Type: CVI_VP_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv66"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
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
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv66"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vuw_vrotr_vuwvuw(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vrotr(vu, vv)
}

/// `Vd32.w=vsatdw(Vu32.w,Vv32.w)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv66"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vsatdw_vwvw(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vsatdw(vu, vv)
}

/// `Vd32.w=vadd(Vu32.w,Vv32.w,Qs4):carry:sat` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv66"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vadd_vwvwq_carry_sat(
    vu: HvxVector,
    vv: HvxVector,
    qs: HvxVectorPred,
) -> HvxVector {
    vaddcarrysat(vu, vv, qs)
}

// ============================================================================
// HVX Architecture v68 intrinsics
// Target feature: hvxv68
// ============================================================================

/// `Vdd32.w=v6mpy(Vuu32.ub,Vvv32.b,#u2):h`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
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
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
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
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
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
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
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
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vhf_vabs_vhf(vu: HvxVector) -> HvxVector {
    vabs_hf(vu)
}

/// `Vd32.sf=vabs(Vu32.sf)`
///
/// Instruction Type: CVI_VX_LATE
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vsf_vabs_vsf(vu: HvxVector) -> HvxVector {
    vabs_sf(vu)
}

/// `Vd32.qf16=vadd(Vu32.hf,Vv32.hf)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vqf16_vadd_vhfvhf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vadd_hf(vu, vv)
}

/// `Vd32.hf=vadd(Vu32.hf,Vv32.hf)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vhf_vadd_vhfvhf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vadd_hf_hf(vu, vv)
}

/// `Vd32.qf16=vadd(Vu32.qf16,Vv32.qf16)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vqf16_vadd_vqf16vqf16(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vadd_qf16(vu, vv)
}

/// `Vd32.qf16=vadd(Vu32.qf16,Vv32.hf)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vqf16_vadd_vqf16vhf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vadd_qf16_mix(vu, vv)
}

/// `Vd32.qf32=vadd(Vu32.qf32,Vv32.qf32)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vqf32_vadd_vqf32vqf32(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vadd_qf32(vu, vv)
}

/// `Vd32.qf32=vadd(Vu32.qf32,Vv32.sf)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vqf32_vadd_vqf32vsf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vadd_qf32_mix(vu, vv)
}

/// `Vd32.qf32=vadd(Vu32.sf,Vv32.sf)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vqf32_vadd_vsfvsf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vadd_sf(vu, vv)
}

/// `Vdd32.sf=vadd(Vu32.hf,Vv32.hf)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wsf_vadd_vhfvhf(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
    vadd_sf_hf(vu, vv)
}

/// `Vd32.sf=vadd(Vu32.sf,Vv32.sf)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vsf_vadd_vsfvsf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vadd_sf_sf(vu, vv)
}

/// `Vd32.w=vfmv(Vu32.w)`
///
/// Instruction Type: CVI_VX_LATE
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vfmv_vw(vu: HvxVector) -> HvxVector {
    vassign_fp(vu)
}

/// `Vd32.hf=Vu32.qf16`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vhf_equals_vqf16(vu: HvxVector) -> HvxVector {
    vconv_hf_qf16(vu)
}

/// `Vd32.hf=Vuu32.qf32`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vhf_equals_wqf32(vuu: HvxVectorPair) -> HvxVector {
    vconv_hf_qf32(vuu)
}

/// `Vd32.sf=Vu32.qf32`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vsf_equals_vqf32(vu: HvxVector) -> HvxVector {
    vconv_sf_qf32(vu)
}

/// `Vd32.b=vcvt(Vu32.hf,Vv32.hf)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vb_vcvt_vhfvhf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vcvt_b_hf(vu, vv)
}

/// `Vd32.h=vcvt(Vu32.hf)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_vcvt_vhf(vu: HvxVector) -> HvxVector {
    vcvt_h_hf(vu)
}

/// `Vdd32.hf=vcvt(Vu32.b)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_whf_vcvt_vb(vu: HvxVector) -> HvxVectorPair {
    vcvt_hf_b(vu)
}

/// `Vd32.hf=vcvt(Vu32.h)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vhf_vcvt_vh(vu: HvxVector) -> HvxVector {
    vcvt_hf_h(vu)
}

/// `Vd32.hf=vcvt(Vu32.sf,Vv32.sf)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vhf_vcvt_vsfvsf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vcvt_hf_sf(vu, vv)
}

/// `Vdd32.hf=vcvt(Vu32.ub)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_whf_vcvt_vub(vu: HvxVector) -> HvxVectorPair {
    vcvt_hf_ub(vu)
}

/// `Vd32.hf=vcvt(Vu32.uh)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vhf_vcvt_vuh(vu: HvxVector) -> HvxVector {
    vcvt_hf_uh(vu)
}

/// `Vdd32.sf=vcvt(Vu32.hf)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wsf_vcvt_vhf(vu: HvxVector) -> HvxVectorPair {
    vcvt_sf_hf(vu)
}

/// `Vd32.ub=vcvt(Vu32.hf,Vv32.hf)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vub_vcvt_vhfvhf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vcvt_ub_hf(vu, vv)
}

/// `Vd32.uh=vcvt(Vu32.hf)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vuh_vcvt_vhf(vu: HvxVector) -> HvxVector {
    vcvt_uh_hf(vu)
}

/// `Vd32.sf=vdmpy(Vu32.hf,Vv32.hf)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vsf_vdmpy_vhfvhf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vdmpy_sf_hf(vu, vv)
}

/// `Vx32.sf+=vdmpy(Vu32.hf,Vv32.hf)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vsf_vdmpyacc_vsfvhfvhf(vx: HvxVector, vu: HvxVector, vv: HvxVector) -> HvxVector {
    vdmpy_sf_hf_acc(vx, vu, vv)
}

/// `Vd32.hf=vfmax(Vu32.hf,Vv32.hf)`
///
/// Instruction Type: CVI_VX_LATE
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vhf_vfmax_vhfvhf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vfmax_hf(vu, vv)
}

/// `Vd32.sf=vfmax(Vu32.sf,Vv32.sf)`
///
/// Instruction Type: CVI_VX_LATE
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vsf_vfmax_vsfvsf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vfmax_sf(vu, vv)
}

/// `Vd32.hf=vfmin(Vu32.hf,Vv32.hf)`
///
/// Instruction Type: CVI_VX_LATE
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vhf_vfmin_vhfvhf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vfmin_hf(vu, vv)
}

/// `Vd32.sf=vfmin(Vu32.sf,Vv32.sf)`
///
/// Instruction Type: CVI_VX_LATE
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vsf_vfmin_vsfvsf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vfmin_sf(vu, vv)
}

/// `Vd32.hf=vfneg(Vu32.hf)`
///
/// Instruction Type: CVI_VX_LATE
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vhf_vfneg_vhf(vu: HvxVector) -> HvxVector {
    vfneg_hf(vu)
}

/// `Vd32.sf=vfneg(Vu32.sf)`
///
/// Instruction Type: CVI_VX_LATE
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vsf_vfneg_vsf(vu: HvxVector) -> HvxVector {
    vfneg_sf(vu)
}

/// `Vd32.hf=vmax(Vu32.hf,Vv32.hf)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vhf_vmax_vhfvhf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmax_hf(vu, vv)
}

/// `Vd32.sf=vmax(Vu32.sf,Vv32.sf)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vsf_vmax_vsfvsf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmax_sf(vu, vv)
}

/// `Vd32.hf=vmin(Vu32.hf,Vv32.hf)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vhf_vmin_vhfvhf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmin_hf(vu, vv)
}

/// `Vd32.sf=vmin(Vu32.sf,Vv32.sf)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vsf_vmin_vsfvsf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmin_sf(vu, vv)
}

/// `Vd32.hf=vmpy(Vu32.hf,Vv32.hf)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vhf_vmpy_vhfvhf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmpy_hf_hf(vu, vv)
}

/// `Vx32.hf+=vmpy(Vu32.hf,Vv32.hf)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vhf_vmpyacc_vhfvhfvhf(vx: HvxVector, vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmpy_hf_hf_acc(vx, vu, vv)
}

/// `Vd32.qf16=vmpy(Vu32.qf16,Vv32.qf16)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vqf16_vmpy_vqf16vqf16(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmpy_qf16(vu, vv)
}

/// `Vd32.qf16=vmpy(Vu32.hf,Vv32.hf)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vqf16_vmpy_vhfvhf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmpy_qf16_hf(vu, vv)
}

/// `Vd32.qf16=vmpy(Vu32.qf16,Vv32.hf)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vqf16_vmpy_vqf16vhf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmpy_qf16_mix_hf(vu, vv)
}

/// `Vd32.qf32=vmpy(Vu32.qf32,Vv32.qf32)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vqf32_vmpy_vqf32vqf32(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmpy_qf32(vu, vv)
}

/// `Vdd32.qf32=vmpy(Vu32.hf,Vv32.hf)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wqf32_vmpy_vhfvhf(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
    vmpy_qf32_hf(vu, vv)
}

/// `Vdd32.qf32=vmpy(Vu32.qf16,Vv32.hf)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wqf32_vmpy_vqf16vhf(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
    vmpy_qf32_mix_hf(vu, vv)
}

/// `Vdd32.qf32=vmpy(Vu32.qf16,Vv32.qf16)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wqf32_vmpy_vqf16vqf16(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
    vmpy_qf32_qf16(vu, vv)
}

/// `Vd32.qf32=vmpy(Vu32.sf,Vv32.sf)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vqf32_vmpy_vsfvsf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmpy_qf32_sf(vu, vv)
}

/// `Vdd32.sf=vmpy(Vu32.hf,Vv32.hf)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wsf_vmpy_vhfvhf(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
    vmpy_sf_hf(vu, vv)
}

/// `Vxx32.sf+=vmpy(Vu32.hf,Vv32.hf)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
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
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vsf_vmpy_vsfvsf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmpy_sf_sf(vu, vv)
}

/// `Vd32.qf16=vsub(Vu32.hf,Vv32.hf)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vqf16_vsub_vhfvhf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vsub_hf(vu, vv)
}

/// `Vd32.hf=vsub(Vu32.hf,Vv32.hf)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vhf_vsub_vhfvhf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vsub_hf_hf(vu, vv)
}

/// `Vd32.qf16=vsub(Vu32.qf16,Vv32.qf16)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vqf16_vsub_vqf16vqf16(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vsub_qf16(vu, vv)
}

/// `Vd32.qf16=vsub(Vu32.qf16,Vv32.hf)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vqf16_vsub_vqf16vhf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vsub_qf16_mix(vu, vv)
}

/// `Vd32.qf32=vsub(Vu32.qf32,Vv32.qf32)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vqf32_vsub_vqf32vqf32(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vsub_qf32(vu, vv)
}

/// `Vd32.qf32=vsub(Vu32.qf32,Vv32.sf)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vqf32_vsub_vqf32vsf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vsub_qf32_mix(vu, vv)
}

/// `Vd32.qf32=vsub(Vu32.sf,Vv32.sf)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vqf32_vsub_vsfvsf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vsub_sf(vu, vv)
}

/// `Vdd32.sf=vsub(Vu32.hf,Vv32.hf)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wsf_vsub_vhfvhf(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
    vsub_sf_hf(vu, vv)
}

/// `Vd32.sf=vsub(Vu32.sf,Vv32.sf)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vsf_vsub_vsfvsf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vsub_sf_sf(vu, vv)
}

/// `Qd4=vcmp.gt(Vu32.hf,Vv32.hf)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_gt_vhfvhf(vu: HvxVector, vv: HvxVector) -> HvxVectorPred {
    vgthf(vu, vv)
}

/// `Qx4&=vcmp.gt(Vu32.hf,Vv32.hf)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_gtand_qvhfvhf(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    vgthf_and(qx, vu, vv)
}

/// `Qx4|=vcmp.gt(Vu32.hf,Vv32.hf)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_gtor_qvhfvhf(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    vgthf_or(qx, vu, vv)
}

/// `Qx4^=vcmp.gt(Vu32.hf,Vv32.hf)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_gtxacc_qvhfvhf(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    vgthf_xor(qx, vu, vv)
}

/// `Qd4=vcmp.gt(Vu32.sf,Vv32.sf)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_gt_vsfvsf(vu: HvxVector, vv: HvxVector) -> HvxVectorPred {
    vgtsf(vu, vv)
}

/// `Qx4&=vcmp.gt(Vu32.sf,Vv32.sf)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_gtand_qvsfvsf(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    vgtsf_and(qx, vu, vv)
}

/// `Qx4|=vcmp.gt(Vu32.sf,Vv32.sf)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_gtor_qvsfvsf(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    vgtsf_or(qx, vu, vv)
}

/// `Qx4^=vcmp.gt(Vu32.sf,Vv32.sf)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv68"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_gtxacc_qvsfvsf(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    vgtsf_xor(qx, vu, vv)
}

// ============================================================================
// HVX Architecture v69 intrinsics
// Target feature: hvxv69
// ============================================================================

/// `Vd32.ub=vasr(Vuu32.uh,Vv32.ub):rnd:sat`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv69"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vub_vasr_wuhvub_rnd_sat(vuu: HvxVectorPair, vv: HvxVector) -> HvxVector {
    vasrvuhubrndsat(vuu, vv)
}

/// `Vd32.ub=vasr(Vuu32.uh,Vv32.ub):sat`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv69"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vub_vasr_wuhvub_sat(vuu: HvxVectorPair, vv: HvxVector) -> HvxVector {
    vasrvuhubsat(vuu, vv)
}

/// `Vd32.uh=vasr(Vuu32.w,Vv32.uh):rnd:sat`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv69"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vuh_vasr_wwvuh_rnd_sat(vuu: HvxVectorPair, vv: HvxVector) -> HvxVector {
    vasrvwuhrndsat(vuu, vv)
}

/// `Vd32.uh=vasr(Vuu32.w,Vv32.uh):sat`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv69"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vuh_vasr_wwvuh_sat(vuu: HvxVectorPair, vv: HvxVector) -> HvxVector {
    vasrvwuhsat(vuu, vv)
}

/// `Vd32.uh=vmpy(Vu32.uh,Vv32.uh):>>16`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv69"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vuh_vmpy_vuhvuh_rs16(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmpyuhvs(vu, vv)
}

// ============================================================================
// HVX Architecture v73 intrinsics
// Target feature: hvxv73
// ============================================================================

/// `Vdd32.sf=vadd(Vu32.bf,Vv32.bf)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv73"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wsf_vadd_vbfvbf(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
    vadd_sf_bf(vu, vv)
}

/// `Vd32.h=Vu32.hf`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv73"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_equals_vhf(vu: HvxVector) -> HvxVector {
    vconv_h_hf(vu)
}

/// `Vd32.hf=Vu32.h`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv73"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vhf_equals_vh(vu: HvxVector) -> HvxVector {
    vconv_hf_h(vu)
}

/// `Vd32.sf=Vu32.w`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv73"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vsf_equals_vw(vu: HvxVector) -> HvxVector {
    vconv_sf_w(vu)
}

/// `Vd32.w=Vu32.sf`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv73"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_equals_vsf(vu: HvxVector) -> HvxVector {
    vconv_w_sf(vu)
}

/// `Vd32.bf=vcvt(Vu32.sf,Vv32.sf)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv73"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vbf_vcvt_vsfvsf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vcvt_bf_sf(vu, vv)
}

/// `Vd32.bf=vmax(Vu32.bf,Vv32.bf)`
///
/// Instruction Type: CVI_VX_LATE
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv73"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vbf_vmax_vbfvbf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmax_bf(vu, vv)
}

/// `Vd32.bf=vmin(Vu32.bf,Vv32.bf)`
///
/// Instruction Type: CVI_VX_LATE
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv73"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vbf_vmin_vbfvbf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmin_bf(vu, vv)
}

/// `Vdd32.sf=vmpy(Vu32.bf,Vv32.bf)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv73"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wsf_vmpy_vbfvbf(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
    vmpy_sf_bf(vu, vv)
}

/// `Vxx32.sf+=vmpy(Vu32.bf,Vv32.bf)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots:
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv73"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wsf_vmpyacc_wsfvbfvbf(
    vxx: HvxVectorPair,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPair {
    vmpy_sf_bf_acc(vxx, vu, vv)
}

/// `Vdd32.sf=vsub(Vu32.bf,Vv32.bf)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv73"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wsf_vsub_vbfvbf(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
    vsub_sf_bf(vu, vv)
}

/// `Qd4=vcmp.gt(Vu32.bf,Vv32.bf)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv73"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_gt_vbfvbf(vu: HvxVector, vv: HvxVector) -> HvxVectorPred {
    vgtbf(vu, vv)
}

/// `Qx4&=vcmp.gt(Vu32.bf,Vv32.bf)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots:
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv73"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_gtand_qvbfvbf(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    vgtbf_and(qx, vu, vv)
}

/// `Qx4|=vcmp.gt(Vu32.bf,Vv32.bf)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots:
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv73"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_gtor_qvbfvbf(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    vgtbf_or(qx, vu, vv)
}

/// `Qx4^=vcmp.gt(Vu32.bf,Vv32.bf)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots:
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv73"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_gtxacc_qvbfvbf(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    vgtbf_xor(qx, vu, vv)
}

// ============================================================================
// HVX Architecture v79 intrinsics
// Target feature: hvxv79
// ============================================================================

/// `Vd32=vgetqfext(Vu32.x,Rt32)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv79"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_v_vgetqfext_vr(vu: HvxVector, rt: i32) -> HvxVector {
    get_qfext(vu, rt)
}

/// `Vx32|=vgetqfext(Vu32.x,Rt32)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv79"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_v_vgetqfextor_vvr(vx: HvxVector, vu: HvxVector, rt: i32) -> HvxVector {
    get_qfext_oracc(vx, vu, rt)
}

/// `Vd32.x=vsetqfext(Vu32,Rt32)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv79"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_v_vsetqfext_vr(vu: HvxVector, rt: i32) -> HvxVector {
    set_qfext(vu, rt)
}

/// `Vd32.f8=vabs(Vu32.f8)`
///
/// Instruction Type: CVI_VX_LATE
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv79"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_v_vabs_v(vu: HvxVector) -> HvxVector {
    vabs_f8(vu)
}

/// `Vdd32.hf=vadd(Vu32.f8,Vv32.f8)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv79"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_whf_vadd_vv(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
    vadd_hf_f8(vu, vv)
}

/// `Vd32.b=vcvt2(Vu32.hf,Vv32.hf)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv79"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vb_vcvt2_vhfvhf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vcvt2_b_hf(vu, vv)
}

/// `Vdd32.hf=vcvt2(Vu32.b)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv79"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_whf_vcvt2_vb(vu: HvxVector) -> HvxVectorPair {
    vcvt2_hf_b(vu)
}

/// `Vdd32.hf=vcvt2(Vu32.ub)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv79"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_whf_vcvt2_vub(vu: HvxVector) -> HvxVectorPair {
    vcvt2_hf_ub(vu)
}

/// `Vd32.ub=vcvt2(Vu32.hf,Vv32.hf)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv79"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vub_vcvt2_vhfvhf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vcvt2_ub_hf(vu, vv)
}

/// `Vd32.f8=vcvt(Vu32.hf,Vv32.hf)`
///
/// Instruction Type: CVI_VX
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv79"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_v_vcvt_vhfvhf(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vcvt_f8_hf(vu, vv)
}

/// `Vdd32.hf=vcvt(Vu32.f8)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv79"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_whf_vcvt_v(vu: HvxVector) -> HvxVectorPair {
    vcvt_hf_f8(vu)
}

/// `Vd32.f8=vfmax(Vu32.f8,Vv32.f8)`
///
/// Instruction Type: CVI_VX_LATE
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv79"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_v_vfmax_vv(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vfmax_f8(vu, vv)
}

/// `Vd32.f8=vfmin(Vu32.f8,Vv32.f8)`
///
/// Instruction Type: CVI_VX_LATE
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv79"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_v_vfmin_vv(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vfmin_f8(vu, vv)
}

/// `Vd32.f8=vfneg(Vu32.f8)`
///
/// Instruction Type: CVI_VX_LATE
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv79"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_v_vfneg_v(vu: HvxVector) -> HvxVector {
    vfneg_f8(vu)
}

/// `Vd32=vmerge(Vu32.x,Vv32.w)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv79"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_v_vmerge_vvw(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vmerge_qf(vu, vv)
}

/// `Vdd32.hf=vmpy(Vu32.f8,Vv32.f8)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv79"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_whf_vmpy_vv(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
    vmpy_hf_f8(vu, vv)
}

/// `Vxx32.hf+=vmpy(Vu32.f8,Vv32.f8)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots:
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv79"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_whf_vmpyacc_whfvv(
    vxx: HvxVectorPair,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPair {
    vmpy_hf_f8_acc(vxx, vu, vv)
}

/// `Vd32.qf16=vmpy(Vu32.hf,Rt32.hf)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv79"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vqf16_vmpy_vhfrhf(vu: HvxVector, rt: i32) -> HvxVector {
    vmpy_rt_hf(vu, rt)
}

/// `Vd32.qf16=vmpy(Vu32.qf16,Rt32.hf)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv79"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vqf16_vmpy_vqf16rhf(vu: HvxVector, rt: i32) -> HvxVector {
    vmpy_rt_qf16(vu, rt)
}

/// `Vd32.qf32=vmpy(Vu32.sf,Rt32.sf)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv79"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vqf32_vmpy_vsfrsf(vu: HvxVector, rt: i32) -> HvxVector {
    vmpy_rt_sf(vu, rt)
}

/// `Vdd32.hf=vsub(Vu32.f8,Vv32.f8)`
///
/// Instruction Type: CVI_VX_DV
/// Execution Slots: SLOT23
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv79"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_whf_vsub_vv(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
    vsub_hf_f8(vu, vv)
}

// ============================================================================
// HVX Architecture v81 intrinsics
// Target feature: hvxv81
// ============================================================================

/// `Vd32.qf16=vabs(Vu32.hf)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv81"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vqf16_vabs_vhf(vu: HvxVector) -> HvxVector {
    vabs_qf16_hf(vu)
}

/// `Vd32.qf16=vabs(Vu32.qf16)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv81"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vqf16_vabs_vqf16(vu: HvxVector) -> HvxVector {
    vabs_qf16_qf16(vu)
}

/// `Vd32.qf32=vabs(Vu32.qf32)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv81"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vqf32_vabs_vqf32(vu: HvxVector) -> HvxVector {
    vabs_qf32_qf32(vu)
}

/// `Vd32.qf32=vabs(Vu32.sf)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv81"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vqf32_vabs_vsf(vu: HvxVector) -> HvxVector {
    vabs_qf32_sf(vu)
}

/// `Vd32=valign4(Vu32,Vv32,Rt8)`
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv81"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_v_valign4_vvr(vu: HvxVector, vv: HvxVector, rt: i32) -> HvxVector {
    valign4(vu, vv, rt)
}

/// `Vd32.bf=Vuu32.qf32`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv81"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vbf_equals_wqf32(vuu: HvxVectorPair) -> HvxVector {
    vconv_bf_qf32(vuu)
}

/// `Vd32.f8=Vu32.qf16`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv81"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_v_equals_vqf16(vu: HvxVector) -> HvxVector {
    vconv_f8_qf16(vu)
}

/// `Vd32.h=Vu32.hf:rnd`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv81"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vh_equals_vhf_rnd(vu: HvxVector) -> HvxVector {
    vconv_h_hf_rnd(vu)
}

/// `Vdd32.qf16=Vu32.f8`
///
/// Instruction Type: CVI_VP_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv81"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_wqf16_equals_v(vu: HvxVector) -> HvxVectorPair {
    vconv_qf16_f8(vu)
}

/// `Vd32.qf16=Vu32.hf`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv81"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vqf16_equals_vhf(vu: HvxVector) -> HvxVector {
    vconv_qf16_hf(vu)
}

/// `Vd32.qf16=Vu32.qf16`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv81"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vqf16_equals_vqf16(vu: HvxVector) -> HvxVector {
    vconv_qf16_qf16(vu)
}

/// `Vd32.qf32=Vu32.qf32`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv81"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vqf32_equals_vqf32(vu: HvxVector) -> HvxVector {
    vconv_qf32_qf32(vu)
}

/// `Vd32.qf32=Vu32.sf`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv81"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vqf32_equals_vsf(vu: HvxVector) -> HvxVector {
    vconv_qf32_sf(vu)
}

/// `Vd32.w=vilog2(Vu32.hf)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv81"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vilog2_vhf(vu: HvxVector) -> HvxVector {
    vilog2_hf(vu)
}

/// `Vd32.w=vilog2(Vu32.qf16)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv81"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vilog2_vqf16(vu: HvxVector) -> HvxVector {
    vilog2_qf16(vu)
}

/// `Vd32.w=vilog2(Vu32.qf32)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv81"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vilog2_vqf32(vu: HvxVector) -> HvxVector {
    vilog2_qf32(vu)
}

/// `Vd32.w=vilog2(Vu32.sf)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv81"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vw_vilog2_vsf(vu: HvxVector) -> HvxVector {
    vilog2_sf(vu)
}

/// `Vd32.qf16=vneg(Vu32.hf)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv81"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vqf16_vneg_vhf(vu: HvxVector) -> HvxVector {
    vneg_qf16_hf(vu)
}

/// `Vd32.qf16=vneg(Vu32.qf16)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv81"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vqf16_vneg_vqf16(vu: HvxVector) -> HvxVector {
    vneg_qf16_qf16(vu)
}

/// `Vd32.qf32=vneg(Vu32.qf32)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv81"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vqf32_vneg_vqf32(vu: HvxVector) -> HvxVector {
    vneg_qf32_qf32(vu)
}

/// `Vd32.qf32=vneg(Vu32.sf)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv81"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vqf32_vneg_vsf(vu: HvxVector) -> HvxVector {
    vneg_qf32_sf(vu)
}

/// `Vd32.qf16=vsub(Vu32.hf,Vv32.qf16)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv81"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vqf16_vsub_vhfvqf16(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vsub_hf_mix(vu, vv)
}

/// `Vd32.qf32=vsub(Vu32.sf,Vv32.qf32)`
///
/// Instruction Type: CVI_VS
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv81"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_vqf32_vsub_vsfvqf32(vu: HvxVector, vv: HvxVector) -> HvxVector {
    vsub_sf_mix(vu, vv)
}

/// `Qd4=vcmp.eq(Vu32.hf,Vv32.hf)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv81"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_eq_vhfvhf(vu: HvxVector, vv: HvxVector) -> HvxVectorPred {
    veqhf(vu, vv)
}

/// `Qx4&=vcmp.eq(Vu32.hf,Vv32.hf)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots:
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv81"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_eqand_qvhfvhf(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    veqhf_and(qx, vu, vv)
}

/// `Qx4|=vcmp.eq(Vu32.hf,Vv32.hf)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots:
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv81"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_eqor_qvhfvhf(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    veqhf_or(qx, vu, vv)
}

/// `Qx4^=vcmp.eq(Vu32.hf,Vv32.hf)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots:
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv81"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_eqxacc_qvhfvhf(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    veqhf_xor(qx, vu, vv)
}

/// `Qd4=vcmp.eq(Vu32.sf,Vv32.sf)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots: SLOT0123
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv81"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_eq_vsfvsf(vu: HvxVector, vv: HvxVector) -> HvxVectorPred {
    veqsf(vu, vv)
}

/// `Qx4&=vcmp.eq(Vu32.sf,Vv32.sf)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots:
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv81"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_eqand_qvsfvsf(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    veqsf_and(qx, vu, vv)
}

/// `Qx4|=vcmp.eq(Vu32.sf,Vv32.sf)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots:
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv81"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_eqor_qvsfvsf(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    veqsf_or(qx, vu, vv)
}

/// `Qx4^=vcmp.eq(Vu32.sf,Vv32.sf)` (compound operation)
///
/// Instruction Type: CVI_VA
/// Execution Slots:
#[inline]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "hvxv81"))]
#[unstable(feature = "stdarch_hexagon", issue = "none")]
pub unsafe fn q6_q_vcmp_eqxacc_qvsfvsf(
    qx: HvxVectorPred,
    vu: HvxVector,
    vv: HvxVector,
) -> HvxVectorPred {
    veqsf_xor(qx, vu, vv)
}

// To test compilation, run:
// NORUN=1 NOSTD=1 TARGET=hexagon-unknown-linux-musl CARGO_UNSTABLE_BUILD_STD=core ci/run.sh
//
// Note: Actual execution requires HVX-capable Hexagon hardware.
// These tests primarily verify compile-time correctness and type signatures.
#[cfg(test)]
mod tests {
    use super::*;

    // Basic compile tests - verify type sizes
    #[test]
    fn test_types_exist() {
        // Verify type sizes for 128-byte HVX mode
        assert_eq!(core::mem::size_of::<HvxVector>(), 128);
        assert_eq!(core::mem::size_of::<HvxVectorPair>(), 256);
        assert_eq!(core::mem::size_of::<HvxVectorPred>(), 128);
    }

    #[test]
    fn test_type_alignment() {
        // HVX vectors should be properly aligned
        assert!(core::mem::align_of::<HvxVector>() >= 128);
        assert!(core::mem::align_of::<HvxVectorPair>() >= 128);
        assert!(core::mem::align_of::<HvxVectorPred>() >= 128);
    }

    // These no_mangle functions are used to verify the intrinsics compile
    // to the expected assembly instructions. Run with --emit=asm to verify.
    #[unsafe(no_mangle)]
    unsafe fn test_q6_v_hi_w(vss: HvxVectorPair) -> HvxVector {
        q6_v_hi_w(vss)
    }

    #[unsafe(no_mangle)]
    unsafe fn test_q6_v_lo_w(vss: HvxVectorPair) -> HvxVector {
        q6_v_lo_w(vss)
    }

    #[unsafe(no_mangle)]
    unsafe fn test_q6_v_vsplat_r(rt: i32) -> HvxVector {
        q6_v_vsplat_r(rt)
    }

    #[unsafe(no_mangle)]
    unsafe fn test_q6_vb_vadd_vbvb(vu: HvxVector, vv: HvxVector) -> HvxVector {
        q6_vb_vadd_vbvb(vu, vv)
    }

    #[unsafe(no_mangle)]
    unsafe fn test_q6_vh_vadd_vhvh(vu: HvxVector, vv: HvxVector) -> HvxVector {
        q6_vh_vadd_vhvh(vu, vv)
    }

    #[unsafe(no_mangle)]
    unsafe fn test_q6_vw_vadd_vwvw(vu: HvxVector, vv: HvxVector) -> HvxVector {
        q6_vw_vadd_vwvw(vu, vv)
    }

    #[unsafe(no_mangle)]
    unsafe fn test_q6_w_vcombine_vv(vu: HvxVector, vv: HvxVector) -> HvxVectorPair {
        q6_w_vcombine_vv(vu, vv)
    }

    #[unsafe(no_mangle)]
    unsafe fn test_q6_v_vzero() -> HvxVector {
        q6_v_vzero()
    }
}
