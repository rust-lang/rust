//! Hexagon scalar intrinsics
//!
//! This module provides intrinsics for scalar (non-HVX) Hexagon DSP operations,
//! including arithmetic, multiply, shift, saturate, compare, and floating-point
//! operations.
//!
//! [Hexagon V68 Programmer's Reference Manual](https://docs.qualcomm.com/doc/80-N2040-45)
//!
//! ## Naming Convention
//!
//! Function names preserve the original Q6 naming case because the convention
//! uses case to distinguish register types:
//! - `P` (uppercase) = 64-bit register pair (`Word64`)
//! - `p` (lowercase) = predicate register (`Byte`)
//!
//! For example, `Q6_P_and_PP` operates on 64-bit pairs while `Q6_p_and_pp`
//! operates on predicate registers.
//!
//! ## Architecture Versions
//!
//! Most scalar intrinsics are available on all Hexagon architectures.
//! Some intrinsics require specific architecture versions (v60, v62, v65,
//! v66, v67, v68, or v67+audio) and carry
//! `#[target_feature(enable = "v68")]` (or the appropriate version).
//! Enable these with `-C target-feature=+v68` or by setting the target CPU
//! via `-C target-cpu=hexagonv68`.
//!
//! Each version includes all features from previous versions.

#![allow(non_snake_case)]

#[cfg(test)]
use stdarch_test::assert_instr;

// LLVM intrinsic declarations for Hexagon scalar operations
#[allow(improper_ctypes)]
unsafe extern "unadjusted" {
    #[link_name = "llvm.hexagon.A2.abs"]
    fn hexagon_A2_abs(_: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.absp"]
    fn hexagon_A2_absp(_: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.abssat"]
    fn hexagon_A2_abssat(_: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.add"]
    fn hexagon_A2_add(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.addh.h16.hh"]
    fn hexagon_A2_addh_h16_hh(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.addh.h16.hl"]
    fn hexagon_A2_addh_h16_hl(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.addh.h16.lh"]
    fn hexagon_A2_addh_h16_lh(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.addh.h16.ll"]
    fn hexagon_A2_addh_h16_ll(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.addh.h16.sat.hh"]
    fn hexagon_A2_addh_h16_sat_hh(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.addh.h16.sat.hl"]
    fn hexagon_A2_addh_h16_sat_hl(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.addh.h16.sat.lh"]
    fn hexagon_A2_addh_h16_sat_lh(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.addh.h16.sat.ll"]
    fn hexagon_A2_addh_h16_sat_ll(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.addh.l16.hl"]
    fn hexagon_A2_addh_l16_hl(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.addh.l16.ll"]
    fn hexagon_A2_addh_l16_ll(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.addh.l16.sat.hl"]
    fn hexagon_A2_addh_l16_sat_hl(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.addh.l16.sat.ll"]
    fn hexagon_A2_addh_l16_sat_ll(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.addi"]
    fn hexagon_A2_addi(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.addp"]
    fn hexagon_A2_addp(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.addpsat"]
    fn hexagon_A2_addpsat(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.addsat"]
    fn hexagon_A2_addsat(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.addsp"]
    fn hexagon_A2_addsp(_: i32, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.and"]
    fn hexagon_A2_and(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.andir"]
    fn hexagon_A2_andir(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.andp"]
    fn hexagon_A2_andp(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.aslh"]
    fn hexagon_A2_aslh(_: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.asrh"]
    fn hexagon_A2_asrh(_: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.combine.hh"]
    fn hexagon_A2_combine_hh(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.combine.hl"]
    fn hexagon_A2_combine_hl(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.combine.lh"]
    fn hexagon_A2_combine_lh(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.combine.ll"]
    fn hexagon_A2_combine_ll(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.combineii"]
    fn hexagon_A2_combineii(_: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.A2.combinew"]
    fn hexagon_A2_combinew(_: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.A2.max"]
    fn hexagon_A2_max(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.maxp"]
    fn hexagon_A2_maxp(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.maxu"]
    fn hexagon_A2_maxu(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.maxup"]
    fn hexagon_A2_maxup(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.min"]
    fn hexagon_A2_min(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.minp"]
    fn hexagon_A2_minp(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.minu"]
    fn hexagon_A2_minu(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.minup"]
    fn hexagon_A2_minup(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.neg"]
    fn hexagon_A2_neg(_: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.negp"]
    fn hexagon_A2_negp(_: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.negsat"]
    fn hexagon_A2_negsat(_: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.not"]
    fn hexagon_A2_not(_: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.notp"]
    fn hexagon_A2_notp(_: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.or"]
    fn hexagon_A2_or(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.orir"]
    fn hexagon_A2_orir(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.orp"]
    fn hexagon_A2_orp(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.roundsat"]
    fn hexagon_A2_roundsat(_: i64) -> i32;
    #[link_name = "llvm.hexagon.A2.sat"]
    fn hexagon_A2_sat(_: i64) -> i32;
    #[link_name = "llvm.hexagon.A2.satb"]
    fn hexagon_A2_satb(_: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.sath"]
    fn hexagon_A2_sath(_: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.satub"]
    fn hexagon_A2_satub(_: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.satuh"]
    fn hexagon_A2_satuh(_: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.sub"]
    fn hexagon_A2_sub(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.subh.h16.hh"]
    fn hexagon_A2_subh_h16_hh(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.subh.h16.hl"]
    fn hexagon_A2_subh_h16_hl(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.subh.h16.lh"]
    fn hexagon_A2_subh_h16_lh(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.subh.h16.ll"]
    fn hexagon_A2_subh_h16_ll(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.subh.h16.sat.hh"]
    fn hexagon_A2_subh_h16_sat_hh(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.subh.h16.sat.hl"]
    fn hexagon_A2_subh_h16_sat_hl(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.subh.h16.sat.lh"]
    fn hexagon_A2_subh_h16_sat_lh(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.subh.h16.sat.ll"]
    fn hexagon_A2_subh_h16_sat_ll(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.subh.l16.hl"]
    fn hexagon_A2_subh_l16_hl(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.subh.l16.ll"]
    fn hexagon_A2_subh_l16_ll(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.subh.l16.sat.hl"]
    fn hexagon_A2_subh_l16_sat_hl(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.subh.l16.sat.ll"]
    fn hexagon_A2_subh_l16_sat_ll(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.subp"]
    fn hexagon_A2_subp(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.subri"]
    fn hexagon_A2_subri(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.subsat"]
    fn hexagon_A2_subsat(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.svaddh"]
    fn hexagon_A2_svaddh(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.svaddhs"]
    fn hexagon_A2_svaddhs(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.svadduhs"]
    fn hexagon_A2_svadduhs(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.svavgh"]
    fn hexagon_A2_svavgh(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.svavghs"]
    fn hexagon_A2_svavghs(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.svnavgh"]
    fn hexagon_A2_svnavgh(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.svsubh"]
    fn hexagon_A2_svsubh(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.svsubhs"]
    fn hexagon_A2_svsubhs(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.svsubuhs"]
    fn hexagon_A2_svsubuhs(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.swiz"]
    fn hexagon_A2_swiz(_: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.sxtb"]
    fn hexagon_A2_sxtb(_: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.sxth"]
    fn hexagon_A2_sxth(_: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.sxtw"]
    fn hexagon_A2_sxtw(_: i32) -> i64;
    #[link_name = "llvm.hexagon.A2.tfr"]
    fn hexagon_A2_tfr(_: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.tfrih"]
    fn hexagon_A2_tfrih(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.tfril"]
    fn hexagon_A2_tfril(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.tfrp"]
    fn hexagon_A2_tfrp(_: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.tfrpi"]
    fn hexagon_A2_tfrpi(_: i32) -> i64;
    #[link_name = "llvm.hexagon.A2.tfrsi"]
    fn hexagon_A2_tfrsi(_: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.vabsh"]
    fn hexagon_A2_vabsh(_: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vabshsat"]
    fn hexagon_A2_vabshsat(_: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vabsw"]
    fn hexagon_A2_vabsw(_: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vabswsat"]
    fn hexagon_A2_vabswsat(_: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vaddb.map"]
    fn hexagon_A2_vaddb_map(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vaddh"]
    fn hexagon_A2_vaddh(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vaddhs"]
    fn hexagon_A2_vaddhs(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vaddub"]
    fn hexagon_A2_vaddub(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vaddubs"]
    fn hexagon_A2_vaddubs(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vadduhs"]
    fn hexagon_A2_vadduhs(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vaddw"]
    fn hexagon_A2_vaddw(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vaddws"]
    fn hexagon_A2_vaddws(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vavgh"]
    fn hexagon_A2_vavgh(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vavghcr"]
    fn hexagon_A2_vavghcr(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vavghr"]
    fn hexagon_A2_vavghr(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vavgub"]
    fn hexagon_A2_vavgub(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vavgubr"]
    fn hexagon_A2_vavgubr(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vavguh"]
    fn hexagon_A2_vavguh(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vavguhr"]
    fn hexagon_A2_vavguhr(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vavguw"]
    fn hexagon_A2_vavguw(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vavguwr"]
    fn hexagon_A2_vavguwr(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vavgw"]
    fn hexagon_A2_vavgw(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vavgwcr"]
    fn hexagon_A2_vavgwcr(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vavgwr"]
    fn hexagon_A2_vavgwr(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vcmpbeq"]
    fn hexagon_A2_vcmpbeq(_: i64, _: i64) -> i32;
    #[link_name = "llvm.hexagon.A2.vcmpbgtu"]
    fn hexagon_A2_vcmpbgtu(_: i64, _: i64) -> i32;
    #[link_name = "llvm.hexagon.A2.vcmpheq"]
    fn hexagon_A2_vcmpheq(_: i64, _: i64) -> i32;
    #[link_name = "llvm.hexagon.A2.vcmphgt"]
    fn hexagon_A2_vcmphgt(_: i64, _: i64) -> i32;
    #[link_name = "llvm.hexagon.A2.vcmphgtu"]
    fn hexagon_A2_vcmphgtu(_: i64, _: i64) -> i32;
    #[link_name = "llvm.hexagon.A2.vcmpweq"]
    fn hexagon_A2_vcmpweq(_: i64, _: i64) -> i32;
    #[link_name = "llvm.hexagon.A2.vcmpwgt"]
    fn hexagon_A2_vcmpwgt(_: i64, _: i64) -> i32;
    #[link_name = "llvm.hexagon.A2.vcmpwgtu"]
    fn hexagon_A2_vcmpwgtu(_: i64, _: i64) -> i32;
    #[link_name = "llvm.hexagon.A2.vconj"]
    fn hexagon_A2_vconj(_: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vmaxb"]
    fn hexagon_A2_vmaxb(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vmaxh"]
    fn hexagon_A2_vmaxh(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vmaxub"]
    fn hexagon_A2_vmaxub(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vmaxuh"]
    fn hexagon_A2_vmaxuh(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vmaxuw"]
    fn hexagon_A2_vmaxuw(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vmaxw"]
    fn hexagon_A2_vmaxw(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vminb"]
    fn hexagon_A2_vminb(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vminh"]
    fn hexagon_A2_vminh(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vminub"]
    fn hexagon_A2_vminub(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vminuh"]
    fn hexagon_A2_vminuh(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vminuw"]
    fn hexagon_A2_vminuw(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vminw"]
    fn hexagon_A2_vminw(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vnavgh"]
    fn hexagon_A2_vnavgh(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vnavghcr"]
    fn hexagon_A2_vnavghcr(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vnavghr"]
    fn hexagon_A2_vnavghr(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vnavgw"]
    fn hexagon_A2_vnavgw(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vnavgwcr"]
    fn hexagon_A2_vnavgwcr(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vnavgwr"]
    fn hexagon_A2_vnavgwr(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vraddub"]
    fn hexagon_A2_vraddub(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vraddub.acc"]
    fn hexagon_A2_vraddub_acc(_: i64, _: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vrsadub"]
    fn hexagon_A2_vrsadub(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vrsadub.acc"]
    fn hexagon_A2_vrsadub_acc(_: i64, _: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vsubb.map"]
    fn hexagon_A2_vsubb_map(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vsubh"]
    fn hexagon_A2_vsubh(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vsubhs"]
    fn hexagon_A2_vsubhs(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vsubub"]
    fn hexagon_A2_vsubub(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vsububs"]
    fn hexagon_A2_vsububs(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vsubuhs"]
    fn hexagon_A2_vsubuhs(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vsubw"]
    fn hexagon_A2_vsubw(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.vsubws"]
    fn hexagon_A2_vsubws(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.xor"]
    fn hexagon_A2_xor(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.xorp"]
    fn hexagon_A2_xorp(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A2.zxtb"]
    fn hexagon_A2_zxtb(_: i32) -> i32;
    #[link_name = "llvm.hexagon.A2.zxth"]
    fn hexagon_A2_zxth(_: i32) -> i32;
    #[link_name = "llvm.hexagon.A4.andn"]
    fn hexagon_A4_andn(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A4.andnp"]
    fn hexagon_A4_andnp(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A4.bitsplit"]
    fn hexagon_A4_bitsplit(_: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.A4.bitspliti"]
    fn hexagon_A4_bitspliti(_: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.A4.boundscheck"]
    fn hexagon_A4_boundscheck(_: i32, _: i64) -> i32;
    #[link_name = "llvm.hexagon.A4.cmpbeq"]
    fn hexagon_A4_cmpbeq(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A4.cmpbeqi"]
    fn hexagon_A4_cmpbeqi(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A4.cmpbgt"]
    fn hexagon_A4_cmpbgt(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A4.cmpbgti"]
    fn hexagon_A4_cmpbgti(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A4.cmpbgtu"]
    fn hexagon_A4_cmpbgtu(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A4.cmpbgtui"]
    fn hexagon_A4_cmpbgtui(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A4.cmpheq"]
    fn hexagon_A4_cmpheq(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A4.cmpheqi"]
    fn hexagon_A4_cmpheqi(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A4.cmphgt"]
    fn hexagon_A4_cmphgt(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A4.cmphgti"]
    fn hexagon_A4_cmphgti(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A4.cmphgtu"]
    fn hexagon_A4_cmphgtu(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A4.cmphgtui"]
    fn hexagon_A4_cmphgtui(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A4.combineir"]
    fn hexagon_A4_combineir(_: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.A4.combineri"]
    fn hexagon_A4_combineri(_: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.A4.cround.ri"]
    fn hexagon_A4_cround_ri(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A4.cround.rr"]
    fn hexagon_A4_cround_rr(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A4.modwrapu"]
    fn hexagon_A4_modwrapu(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A4.orn"]
    fn hexagon_A4_orn(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A4.ornp"]
    fn hexagon_A4_ornp(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A4.rcmpeq"]
    fn hexagon_A4_rcmpeq(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A4.rcmpeqi"]
    fn hexagon_A4_rcmpeqi(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A4.rcmpneq"]
    fn hexagon_A4_rcmpneq(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A4.rcmpneqi"]
    fn hexagon_A4_rcmpneqi(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A4.round.ri"]
    fn hexagon_A4_round_ri(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A4.round.ri.sat"]
    fn hexagon_A4_round_ri_sat(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A4.round.rr"]
    fn hexagon_A4_round_rr(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A4.round.rr.sat"]
    fn hexagon_A4_round_rr_sat(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A4.tlbmatch"]
    fn hexagon_A4_tlbmatch(_: i64, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A4.vcmpbeq.any"]
    fn hexagon_A4_vcmpbeq_any(_: i64, _: i64) -> i32;
    #[link_name = "llvm.hexagon.A4.vcmpbeqi"]
    fn hexagon_A4_vcmpbeqi(_: i64, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A4.vcmpbgt"]
    fn hexagon_A4_vcmpbgt(_: i64, _: i64) -> i32;
    #[link_name = "llvm.hexagon.A4.vcmpbgti"]
    fn hexagon_A4_vcmpbgti(_: i64, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A4.vcmpbgtui"]
    fn hexagon_A4_vcmpbgtui(_: i64, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A4.vcmpheqi"]
    fn hexagon_A4_vcmpheqi(_: i64, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A4.vcmphgti"]
    fn hexagon_A4_vcmphgti(_: i64, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A4.vcmphgtui"]
    fn hexagon_A4_vcmphgtui(_: i64, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A4.vcmpweqi"]
    fn hexagon_A4_vcmpweqi(_: i64, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A4.vcmpwgti"]
    fn hexagon_A4_vcmpwgti(_: i64, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A4.vcmpwgtui"]
    fn hexagon_A4_vcmpwgtui(_: i64, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A4.vrmaxh"]
    fn hexagon_A4_vrmaxh(_: i64, _: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.A4.vrmaxuh"]
    fn hexagon_A4_vrmaxuh(_: i64, _: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.A4.vrmaxuw"]
    fn hexagon_A4_vrmaxuw(_: i64, _: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.A4.vrmaxw"]
    fn hexagon_A4_vrmaxw(_: i64, _: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.A4.vrminh"]
    fn hexagon_A4_vrminh(_: i64, _: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.A4.vrminuh"]
    fn hexagon_A4_vrminuh(_: i64, _: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.A4.vrminuw"]
    fn hexagon_A4_vrminuw(_: i64, _: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.A4.vrminw"]
    fn hexagon_A4_vrminw(_: i64, _: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.A5.vaddhubs"]
    fn hexagon_A5_vaddhubs(_: i64, _: i64) -> i32;
    #[link_name = "llvm.hexagon.C2.all8"]
    fn hexagon_C2_all8(_: i32) -> i32;
    #[link_name = "llvm.hexagon.C2.and"]
    fn hexagon_C2_and(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.C2.andn"]
    fn hexagon_C2_andn(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.C2.any8"]
    fn hexagon_C2_any8(_: i32) -> i32;
    #[link_name = "llvm.hexagon.C2.bitsclr"]
    fn hexagon_C2_bitsclr(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.C2.bitsclri"]
    fn hexagon_C2_bitsclri(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.C2.bitsset"]
    fn hexagon_C2_bitsset(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.C2.cmpeq"]
    fn hexagon_C2_cmpeq(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.C2.cmpeqi"]
    fn hexagon_C2_cmpeqi(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.C2.cmpeqp"]
    fn hexagon_C2_cmpeqp(_: i64, _: i64) -> i32;
    #[link_name = "llvm.hexagon.C2.cmpgei"]
    fn hexagon_C2_cmpgei(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.C2.cmpgeui"]
    fn hexagon_C2_cmpgeui(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.C2.cmpgt"]
    fn hexagon_C2_cmpgt(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.C2.cmpgti"]
    fn hexagon_C2_cmpgti(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.C2.cmpgtp"]
    fn hexagon_C2_cmpgtp(_: i64, _: i64) -> i32;
    #[link_name = "llvm.hexagon.C2.cmpgtu"]
    fn hexagon_C2_cmpgtu(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.C2.cmpgtui"]
    fn hexagon_C2_cmpgtui(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.C2.cmpgtup"]
    fn hexagon_C2_cmpgtup(_: i64, _: i64) -> i32;
    #[link_name = "llvm.hexagon.C2.cmplt"]
    fn hexagon_C2_cmplt(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.C2.cmpltu"]
    fn hexagon_C2_cmpltu(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.C2.mask"]
    fn hexagon_C2_mask(_: i32) -> i64;
    #[link_name = "llvm.hexagon.C2.mux"]
    fn hexagon_C2_mux(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.C2.muxii"]
    fn hexagon_C2_muxii(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.C2.muxir"]
    fn hexagon_C2_muxir(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.C2.muxri"]
    fn hexagon_C2_muxri(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.C2.not"]
    fn hexagon_C2_not(_: i32) -> i32;
    #[link_name = "llvm.hexagon.C2.or"]
    fn hexagon_C2_or(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.C2.orn"]
    fn hexagon_C2_orn(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.C2.pxfer.map"]
    fn hexagon_C2_pxfer_map(_: i32) -> i32;
    #[link_name = "llvm.hexagon.C2.tfrpr"]
    fn hexagon_C2_tfrpr(_: i32) -> i32;
    #[link_name = "llvm.hexagon.C2.tfrrp"]
    fn hexagon_C2_tfrrp(_: i32) -> i32;
    #[link_name = "llvm.hexagon.C2.vitpack"]
    fn hexagon_C2_vitpack(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.C2.vmux"]
    fn hexagon_C2_vmux(_: i32, _: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.C2.xor"]
    fn hexagon_C2_xor(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.C4.and.and"]
    fn hexagon_C4_and_and(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.C4.and.andn"]
    fn hexagon_C4_and_andn(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.C4.and.or"]
    fn hexagon_C4_and_or(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.C4.and.orn"]
    fn hexagon_C4_and_orn(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.C4.cmplte"]
    fn hexagon_C4_cmplte(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.C4.cmpltei"]
    fn hexagon_C4_cmpltei(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.C4.cmplteu"]
    fn hexagon_C4_cmplteu(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.C4.cmplteui"]
    fn hexagon_C4_cmplteui(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.C4.cmpneq"]
    fn hexagon_C4_cmpneq(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.C4.cmpneqi"]
    fn hexagon_C4_cmpneqi(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.C4.fastcorner9"]
    fn hexagon_C4_fastcorner9(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.C4.fastcorner9.not"]
    fn hexagon_C4_fastcorner9_not(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.C4.nbitsclr"]
    fn hexagon_C4_nbitsclr(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.C4.nbitsclri"]
    fn hexagon_C4_nbitsclri(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.C4.nbitsset"]
    fn hexagon_C4_nbitsset(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.C4.or.and"]
    fn hexagon_C4_or_and(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.C4.or.andn"]
    fn hexagon_C4_or_andn(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.C4.or.or"]
    fn hexagon_C4_or_or(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.C4.or.orn"]
    fn hexagon_C4_or_orn(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.F2.conv.d2df"]
    fn hexagon_F2_conv_d2df(_: i64) -> f64;
    #[link_name = "llvm.hexagon.F2.conv.d2sf"]
    fn hexagon_F2_conv_d2sf(_: i64) -> f32;
    #[link_name = "llvm.hexagon.F2.conv.df2d"]
    fn hexagon_F2_conv_df2d(_: f64) -> i64;
    #[link_name = "llvm.hexagon.F2.conv.df2d.chop"]
    fn hexagon_F2_conv_df2d_chop(_: f64) -> i64;
    #[link_name = "llvm.hexagon.F2.conv.df2sf"]
    fn hexagon_F2_conv_df2sf(_: f64) -> f32;
    #[link_name = "llvm.hexagon.F2.conv.df2ud"]
    fn hexagon_F2_conv_df2ud(_: f64) -> i64;
    #[link_name = "llvm.hexagon.F2.conv.df2ud.chop"]
    fn hexagon_F2_conv_df2ud_chop(_: f64) -> i64;
    #[link_name = "llvm.hexagon.F2.conv.df2uw"]
    fn hexagon_F2_conv_df2uw(_: f64) -> i32;
    #[link_name = "llvm.hexagon.F2.conv.df2uw.chop"]
    fn hexagon_F2_conv_df2uw_chop(_: f64) -> i32;
    #[link_name = "llvm.hexagon.F2.conv.df2w"]
    fn hexagon_F2_conv_df2w(_: f64) -> i32;
    #[link_name = "llvm.hexagon.F2.conv.df2w.chop"]
    fn hexagon_F2_conv_df2w_chop(_: f64) -> i32;
    #[link_name = "llvm.hexagon.F2.conv.sf2d"]
    fn hexagon_F2_conv_sf2d(_: f32) -> i64;
    #[link_name = "llvm.hexagon.F2.conv.sf2d.chop"]
    fn hexagon_F2_conv_sf2d_chop(_: f32) -> i64;
    #[link_name = "llvm.hexagon.F2.conv.sf2df"]
    fn hexagon_F2_conv_sf2df(_: f32) -> f64;
    #[link_name = "llvm.hexagon.F2.conv.sf2ud"]
    fn hexagon_F2_conv_sf2ud(_: f32) -> i64;
    #[link_name = "llvm.hexagon.F2.conv.sf2ud.chop"]
    fn hexagon_F2_conv_sf2ud_chop(_: f32) -> i64;
    #[link_name = "llvm.hexagon.F2.conv.sf2uw"]
    fn hexagon_F2_conv_sf2uw(_: f32) -> i32;
    #[link_name = "llvm.hexagon.F2.conv.sf2uw.chop"]
    fn hexagon_F2_conv_sf2uw_chop(_: f32) -> i32;
    #[link_name = "llvm.hexagon.F2.conv.sf2w"]
    fn hexagon_F2_conv_sf2w(_: f32) -> i32;
    #[link_name = "llvm.hexagon.F2.conv.sf2w.chop"]
    fn hexagon_F2_conv_sf2w_chop(_: f32) -> i32;
    #[link_name = "llvm.hexagon.F2.conv.ud2df"]
    fn hexagon_F2_conv_ud2df(_: i64) -> f64;
    #[link_name = "llvm.hexagon.F2.conv.ud2sf"]
    fn hexagon_F2_conv_ud2sf(_: i64) -> f32;
    #[link_name = "llvm.hexagon.F2.conv.uw2df"]
    fn hexagon_F2_conv_uw2df(_: i32) -> f64;
    #[link_name = "llvm.hexagon.F2.conv.uw2sf"]
    fn hexagon_F2_conv_uw2sf(_: i32) -> f32;
    #[link_name = "llvm.hexagon.F2.conv.w2df"]
    fn hexagon_F2_conv_w2df(_: i32) -> f64;
    #[link_name = "llvm.hexagon.F2.conv.w2sf"]
    fn hexagon_F2_conv_w2sf(_: i32) -> f32;
    #[link_name = "llvm.hexagon.F2.dfclass"]
    fn hexagon_F2_dfclass(_: f64, _: i32) -> i32;
    #[link_name = "llvm.hexagon.F2.dfcmpeq"]
    fn hexagon_F2_dfcmpeq(_: f64, _: f64) -> i32;
    #[link_name = "llvm.hexagon.F2.dfcmpge"]
    fn hexagon_F2_dfcmpge(_: f64, _: f64) -> i32;
    #[link_name = "llvm.hexagon.F2.dfcmpgt"]
    fn hexagon_F2_dfcmpgt(_: f64, _: f64) -> i32;
    #[link_name = "llvm.hexagon.F2.dfcmpuo"]
    fn hexagon_F2_dfcmpuo(_: f64, _: f64) -> i32;
    #[link_name = "llvm.hexagon.F2.dfimm.n"]
    fn hexagon_F2_dfimm_n(_: i32) -> f64;
    #[link_name = "llvm.hexagon.F2.dfimm.p"]
    fn hexagon_F2_dfimm_p(_: i32) -> f64;
    #[link_name = "llvm.hexagon.F2.sfadd"]
    fn hexagon_F2_sfadd(_: f32, _: f32) -> f32;
    #[link_name = "llvm.hexagon.F2.sfclass"]
    fn hexagon_F2_sfclass(_: f32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.F2.sfcmpeq"]
    fn hexagon_F2_sfcmpeq(_: f32, _: f32) -> i32;
    #[link_name = "llvm.hexagon.F2.sfcmpge"]
    fn hexagon_F2_sfcmpge(_: f32, _: f32) -> i32;
    #[link_name = "llvm.hexagon.F2.sfcmpgt"]
    fn hexagon_F2_sfcmpgt(_: f32, _: f32) -> i32;
    #[link_name = "llvm.hexagon.F2.sfcmpuo"]
    fn hexagon_F2_sfcmpuo(_: f32, _: f32) -> i32;
    #[link_name = "llvm.hexagon.F2.sffixupd"]
    fn hexagon_F2_sffixupd(_: f32, _: f32) -> f32;
    #[link_name = "llvm.hexagon.F2.sffixupn"]
    fn hexagon_F2_sffixupn(_: f32, _: f32) -> f32;
    #[link_name = "llvm.hexagon.F2.sffixupr"]
    fn hexagon_F2_sffixupr(_: f32) -> f32;
    #[link_name = "llvm.hexagon.F2.sffma"]
    fn hexagon_F2_sffma(_: f32, _: f32, _: f32) -> f32;
    #[link_name = "llvm.hexagon.F2.sffma.lib"]
    fn hexagon_F2_sffma_lib(_: f32, _: f32, _: f32) -> f32;
    #[link_name = "llvm.hexagon.F2.sffma.sc"]
    fn hexagon_F2_sffma_sc(_: f32, _: f32, _: f32, _: i32) -> f32;
    #[link_name = "llvm.hexagon.F2.sffms"]
    fn hexagon_F2_sffms(_: f32, _: f32, _: f32) -> f32;
    #[link_name = "llvm.hexagon.F2.sffms.lib"]
    fn hexagon_F2_sffms_lib(_: f32, _: f32, _: f32) -> f32;
    #[link_name = "llvm.hexagon.F2.sfimm.n"]
    fn hexagon_F2_sfimm_n(_: i32) -> f32;
    #[link_name = "llvm.hexagon.F2.sfimm.p"]
    fn hexagon_F2_sfimm_p(_: i32) -> f32;
    #[link_name = "llvm.hexagon.F2.sfmax"]
    fn hexagon_F2_sfmax(_: f32, _: f32) -> f32;
    #[link_name = "llvm.hexagon.F2.sfmin"]
    fn hexagon_F2_sfmin(_: f32, _: f32) -> f32;
    #[link_name = "llvm.hexagon.F2.sfmpy"]
    fn hexagon_F2_sfmpy(_: f32, _: f32) -> f32;
    #[link_name = "llvm.hexagon.F2.sfsub"]
    fn hexagon_F2_sfsub(_: f32, _: f32) -> f32;
    #[link_name = "llvm.hexagon.M2.acci"]
    fn hexagon_M2_acci(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.accii"]
    fn hexagon_M2_accii(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.cmaci.s0"]
    fn hexagon_M2_cmaci_s0(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.cmacr.s0"]
    fn hexagon_M2_cmacr_s0(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.cmacs.s0"]
    fn hexagon_M2_cmacs_s0(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.cmacs.s1"]
    fn hexagon_M2_cmacs_s1(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.cmacsc.s0"]
    fn hexagon_M2_cmacsc_s0(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.cmacsc.s1"]
    fn hexagon_M2_cmacsc_s1(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.cmpyi.s0"]
    fn hexagon_M2_cmpyi_s0(_: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.cmpyr.s0"]
    fn hexagon_M2_cmpyr_s0(_: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.cmpyrs.s0"]
    fn hexagon_M2_cmpyrs_s0(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.cmpyrs.s1"]
    fn hexagon_M2_cmpyrs_s1(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.cmpyrsc.s0"]
    fn hexagon_M2_cmpyrsc_s0(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.cmpyrsc.s1"]
    fn hexagon_M2_cmpyrsc_s1(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.cmpys.s0"]
    fn hexagon_M2_cmpys_s0(_: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.cmpys.s1"]
    fn hexagon_M2_cmpys_s1(_: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.cmpysc.s0"]
    fn hexagon_M2_cmpysc_s0(_: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.cmpysc.s1"]
    fn hexagon_M2_cmpysc_s1(_: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.cnacs.s0"]
    fn hexagon_M2_cnacs_s0(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.cnacs.s1"]
    fn hexagon_M2_cnacs_s1(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.cnacsc.s0"]
    fn hexagon_M2_cnacsc_s0(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.cnacsc.s1"]
    fn hexagon_M2_cnacsc_s1(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.dpmpyss.acc.s0"]
    fn hexagon_M2_dpmpyss_acc_s0(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.dpmpyss.nac.s0"]
    fn hexagon_M2_dpmpyss_nac_s0(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.dpmpyss.rnd.s0"]
    fn hexagon_M2_dpmpyss_rnd_s0(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.dpmpyss.s0"]
    fn hexagon_M2_dpmpyss_s0(_: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.dpmpyuu.acc.s0"]
    fn hexagon_M2_dpmpyuu_acc_s0(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.dpmpyuu.nac.s0"]
    fn hexagon_M2_dpmpyuu_nac_s0(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.dpmpyuu.s0"]
    fn hexagon_M2_dpmpyuu_s0(_: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.hmmpyh.rs1"]
    fn hexagon_M2_hmmpyh_rs1(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.hmmpyh.s1"]
    fn hexagon_M2_hmmpyh_s1(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.hmmpyl.rs1"]
    fn hexagon_M2_hmmpyl_rs1(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.hmmpyl.s1"]
    fn hexagon_M2_hmmpyl_s1(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.maci"]
    fn hexagon_M2_maci(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.macsin"]
    fn hexagon_M2_macsin(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.macsip"]
    fn hexagon_M2_macsip(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mmachs.rs0"]
    fn hexagon_M2_mmachs_rs0(_: i64, _: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.mmachs.rs1"]
    fn hexagon_M2_mmachs_rs1(_: i64, _: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.mmachs.s0"]
    fn hexagon_M2_mmachs_s0(_: i64, _: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.mmachs.s1"]
    fn hexagon_M2_mmachs_s1(_: i64, _: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.mmacls.rs0"]
    fn hexagon_M2_mmacls_rs0(_: i64, _: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.mmacls.rs1"]
    fn hexagon_M2_mmacls_rs1(_: i64, _: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.mmacls.s0"]
    fn hexagon_M2_mmacls_s0(_: i64, _: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.mmacls.s1"]
    fn hexagon_M2_mmacls_s1(_: i64, _: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.mmacuhs.rs0"]
    fn hexagon_M2_mmacuhs_rs0(_: i64, _: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.mmacuhs.rs1"]
    fn hexagon_M2_mmacuhs_rs1(_: i64, _: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.mmacuhs.s0"]
    fn hexagon_M2_mmacuhs_s0(_: i64, _: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.mmacuhs.s1"]
    fn hexagon_M2_mmacuhs_s1(_: i64, _: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.mmaculs.rs0"]
    fn hexagon_M2_mmaculs_rs0(_: i64, _: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.mmaculs.rs1"]
    fn hexagon_M2_mmaculs_rs1(_: i64, _: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.mmaculs.s0"]
    fn hexagon_M2_mmaculs_s0(_: i64, _: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.mmaculs.s1"]
    fn hexagon_M2_mmaculs_s1(_: i64, _: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.mmpyh.rs0"]
    fn hexagon_M2_mmpyh_rs0(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.mmpyh.rs1"]
    fn hexagon_M2_mmpyh_rs1(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.mmpyh.s0"]
    fn hexagon_M2_mmpyh_s0(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.mmpyh.s1"]
    fn hexagon_M2_mmpyh_s1(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.mmpyl.rs0"]
    fn hexagon_M2_mmpyl_rs0(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.mmpyl.rs1"]
    fn hexagon_M2_mmpyl_rs1(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.mmpyl.s0"]
    fn hexagon_M2_mmpyl_s0(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.mmpyl.s1"]
    fn hexagon_M2_mmpyl_s1(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.mmpyuh.rs0"]
    fn hexagon_M2_mmpyuh_rs0(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.mmpyuh.rs1"]
    fn hexagon_M2_mmpyuh_rs1(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.mmpyuh.s0"]
    fn hexagon_M2_mmpyuh_s0(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.mmpyuh.s1"]
    fn hexagon_M2_mmpyuh_s1(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.mmpyul.rs0"]
    fn hexagon_M2_mmpyul_rs0(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.mmpyul.rs1"]
    fn hexagon_M2_mmpyul_rs1(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.mmpyul.s0"]
    fn hexagon_M2_mmpyul_s0(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.mmpyul.s1"]
    fn hexagon_M2_mmpyul_s1(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.mpy.acc.hh.s0"]
    fn hexagon_M2_mpy_acc_hh_s0(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.acc.hh.s1"]
    fn hexagon_M2_mpy_acc_hh_s1(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.acc.hl.s0"]
    fn hexagon_M2_mpy_acc_hl_s0(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.acc.hl.s1"]
    fn hexagon_M2_mpy_acc_hl_s1(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.acc.lh.s0"]
    fn hexagon_M2_mpy_acc_lh_s0(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.acc.lh.s1"]
    fn hexagon_M2_mpy_acc_lh_s1(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.acc.ll.s0"]
    fn hexagon_M2_mpy_acc_ll_s0(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.acc.ll.s1"]
    fn hexagon_M2_mpy_acc_ll_s1(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.acc.sat.hh.s0"]
    fn hexagon_M2_mpy_acc_sat_hh_s0(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.acc.sat.hh.s1"]
    fn hexagon_M2_mpy_acc_sat_hh_s1(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.acc.sat.hl.s0"]
    fn hexagon_M2_mpy_acc_sat_hl_s0(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.acc.sat.hl.s1"]
    fn hexagon_M2_mpy_acc_sat_hl_s1(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.acc.sat.lh.s0"]
    fn hexagon_M2_mpy_acc_sat_lh_s0(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.acc.sat.lh.s1"]
    fn hexagon_M2_mpy_acc_sat_lh_s1(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.acc.sat.ll.s0"]
    fn hexagon_M2_mpy_acc_sat_ll_s0(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.acc.sat.ll.s1"]
    fn hexagon_M2_mpy_acc_sat_ll_s1(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.hh.s0"]
    fn hexagon_M2_mpy_hh_s0(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.hh.s1"]
    fn hexagon_M2_mpy_hh_s1(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.hl.s0"]
    fn hexagon_M2_mpy_hl_s0(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.hl.s1"]
    fn hexagon_M2_mpy_hl_s1(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.lh.s0"]
    fn hexagon_M2_mpy_lh_s0(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.lh.s1"]
    fn hexagon_M2_mpy_lh_s1(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.ll.s0"]
    fn hexagon_M2_mpy_ll_s0(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.ll.s1"]
    fn hexagon_M2_mpy_ll_s1(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.nac.hh.s0"]
    fn hexagon_M2_mpy_nac_hh_s0(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.nac.hh.s1"]
    fn hexagon_M2_mpy_nac_hh_s1(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.nac.hl.s0"]
    fn hexagon_M2_mpy_nac_hl_s0(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.nac.hl.s1"]
    fn hexagon_M2_mpy_nac_hl_s1(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.nac.lh.s0"]
    fn hexagon_M2_mpy_nac_lh_s0(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.nac.lh.s1"]
    fn hexagon_M2_mpy_nac_lh_s1(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.nac.ll.s0"]
    fn hexagon_M2_mpy_nac_ll_s0(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.nac.ll.s1"]
    fn hexagon_M2_mpy_nac_ll_s1(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.nac.sat.hh.s0"]
    fn hexagon_M2_mpy_nac_sat_hh_s0(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.nac.sat.hh.s1"]
    fn hexagon_M2_mpy_nac_sat_hh_s1(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.nac.sat.hl.s0"]
    fn hexagon_M2_mpy_nac_sat_hl_s0(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.nac.sat.hl.s1"]
    fn hexagon_M2_mpy_nac_sat_hl_s1(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.nac.sat.lh.s0"]
    fn hexagon_M2_mpy_nac_sat_lh_s0(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.nac.sat.lh.s1"]
    fn hexagon_M2_mpy_nac_sat_lh_s1(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.nac.sat.ll.s0"]
    fn hexagon_M2_mpy_nac_sat_ll_s0(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.nac.sat.ll.s1"]
    fn hexagon_M2_mpy_nac_sat_ll_s1(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.rnd.hh.s0"]
    fn hexagon_M2_mpy_rnd_hh_s0(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.rnd.hh.s1"]
    fn hexagon_M2_mpy_rnd_hh_s1(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.rnd.hl.s0"]
    fn hexagon_M2_mpy_rnd_hl_s0(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.rnd.hl.s1"]
    fn hexagon_M2_mpy_rnd_hl_s1(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.rnd.lh.s0"]
    fn hexagon_M2_mpy_rnd_lh_s0(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.rnd.lh.s1"]
    fn hexagon_M2_mpy_rnd_lh_s1(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.rnd.ll.s0"]
    fn hexagon_M2_mpy_rnd_ll_s0(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.rnd.ll.s1"]
    fn hexagon_M2_mpy_rnd_ll_s1(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.sat.hh.s0"]
    fn hexagon_M2_mpy_sat_hh_s0(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.sat.hh.s1"]
    fn hexagon_M2_mpy_sat_hh_s1(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.sat.hl.s0"]
    fn hexagon_M2_mpy_sat_hl_s0(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.sat.hl.s1"]
    fn hexagon_M2_mpy_sat_hl_s1(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.sat.lh.s0"]
    fn hexagon_M2_mpy_sat_lh_s0(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.sat.lh.s1"]
    fn hexagon_M2_mpy_sat_lh_s1(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.sat.ll.s0"]
    fn hexagon_M2_mpy_sat_ll_s0(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.sat.ll.s1"]
    fn hexagon_M2_mpy_sat_ll_s1(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.sat.rnd.hh.s0"]
    fn hexagon_M2_mpy_sat_rnd_hh_s0(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.sat.rnd.hh.s1"]
    fn hexagon_M2_mpy_sat_rnd_hh_s1(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.sat.rnd.hl.s0"]
    fn hexagon_M2_mpy_sat_rnd_hl_s0(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.sat.rnd.hl.s1"]
    fn hexagon_M2_mpy_sat_rnd_hl_s1(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.sat.rnd.lh.s0"]
    fn hexagon_M2_mpy_sat_rnd_lh_s0(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.sat.rnd.lh.s1"]
    fn hexagon_M2_mpy_sat_rnd_lh_s1(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.sat.rnd.ll.s0"]
    fn hexagon_M2_mpy_sat_rnd_ll_s0(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.sat.rnd.ll.s1"]
    fn hexagon_M2_mpy_sat_rnd_ll_s1(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.up"]
    fn hexagon_M2_mpy_up(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.up.s1"]
    fn hexagon_M2_mpy_up_s1(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpy.up.s1.sat"]
    fn hexagon_M2_mpy_up_s1_sat(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpyd.acc.hh.s0"]
    fn hexagon_M2_mpyd_acc_hh_s0(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyd.acc.hh.s1"]
    fn hexagon_M2_mpyd_acc_hh_s1(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyd.acc.hl.s0"]
    fn hexagon_M2_mpyd_acc_hl_s0(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyd.acc.hl.s1"]
    fn hexagon_M2_mpyd_acc_hl_s1(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyd.acc.lh.s0"]
    fn hexagon_M2_mpyd_acc_lh_s0(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyd.acc.lh.s1"]
    fn hexagon_M2_mpyd_acc_lh_s1(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyd.acc.ll.s0"]
    fn hexagon_M2_mpyd_acc_ll_s0(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyd.acc.ll.s1"]
    fn hexagon_M2_mpyd_acc_ll_s1(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyd.hh.s0"]
    fn hexagon_M2_mpyd_hh_s0(_: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyd.hh.s1"]
    fn hexagon_M2_mpyd_hh_s1(_: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyd.hl.s0"]
    fn hexagon_M2_mpyd_hl_s0(_: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyd.hl.s1"]
    fn hexagon_M2_mpyd_hl_s1(_: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyd.lh.s0"]
    fn hexagon_M2_mpyd_lh_s0(_: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyd.lh.s1"]
    fn hexagon_M2_mpyd_lh_s1(_: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyd.ll.s0"]
    fn hexagon_M2_mpyd_ll_s0(_: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyd.ll.s1"]
    fn hexagon_M2_mpyd_ll_s1(_: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyd.nac.hh.s0"]
    fn hexagon_M2_mpyd_nac_hh_s0(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyd.nac.hh.s1"]
    fn hexagon_M2_mpyd_nac_hh_s1(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyd.nac.hl.s0"]
    fn hexagon_M2_mpyd_nac_hl_s0(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyd.nac.hl.s1"]
    fn hexagon_M2_mpyd_nac_hl_s1(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyd.nac.lh.s0"]
    fn hexagon_M2_mpyd_nac_lh_s0(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyd.nac.lh.s1"]
    fn hexagon_M2_mpyd_nac_lh_s1(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyd.nac.ll.s0"]
    fn hexagon_M2_mpyd_nac_ll_s0(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyd.nac.ll.s1"]
    fn hexagon_M2_mpyd_nac_ll_s1(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyd.rnd.hh.s0"]
    fn hexagon_M2_mpyd_rnd_hh_s0(_: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyd.rnd.hh.s1"]
    fn hexagon_M2_mpyd_rnd_hh_s1(_: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyd.rnd.hl.s0"]
    fn hexagon_M2_mpyd_rnd_hl_s0(_: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyd.rnd.hl.s1"]
    fn hexagon_M2_mpyd_rnd_hl_s1(_: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyd.rnd.lh.s0"]
    fn hexagon_M2_mpyd_rnd_lh_s0(_: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyd.rnd.lh.s1"]
    fn hexagon_M2_mpyd_rnd_lh_s1(_: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyd.rnd.ll.s0"]
    fn hexagon_M2_mpyd_rnd_ll_s0(_: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyd.rnd.ll.s1"]
    fn hexagon_M2_mpyd_rnd_ll_s1(_: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyi"]
    fn hexagon_M2_mpyi(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpysmi"]
    fn hexagon_M2_mpysmi(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpysu.up"]
    fn hexagon_M2_mpysu_up(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpyu.acc.hh.s0"]
    fn hexagon_M2_mpyu_acc_hh_s0(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpyu.acc.hh.s1"]
    fn hexagon_M2_mpyu_acc_hh_s1(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpyu.acc.hl.s0"]
    fn hexagon_M2_mpyu_acc_hl_s0(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpyu.acc.hl.s1"]
    fn hexagon_M2_mpyu_acc_hl_s1(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpyu.acc.lh.s0"]
    fn hexagon_M2_mpyu_acc_lh_s0(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpyu.acc.lh.s1"]
    fn hexagon_M2_mpyu_acc_lh_s1(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpyu.acc.ll.s0"]
    fn hexagon_M2_mpyu_acc_ll_s0(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpyu.acc.ll.s1"]
    fn hexagon_M2_mpyu_acc_ll_s1(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpyu.hh.s0"]
    fn hexagon_M2_mpyu_hh_s0(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpyu.hh.s1"]
    fn hexagon_M2_mpyu_hh_s1(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpyu.hl.s0"]
    fn hexagon_M2_mpyu_hl_s0(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpyu.hl.s1"]
    fn hexagon_M2_mpyu_hl_s1(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpyu.lh.s0"]
    fn hexagon_M2_mpyu_lh_s0(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpyu.lh.s1"]
    fn hexagon_M2_mpyu_lh_s1(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpyu.ll.s0"]
    fn hexagon_M2_mpyu_ll_s0(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpyu.ll.s1"]
    fn hexagon_M2_mpyu_ll_s1(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpyu.nac.hh.s0"]
    fn hexagon_M2_mpyu_nac_hh_s0(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpyu.nac.hh.s1"]
    fn hexagon_M2_mpyu_nac_hh_s1(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpyu.nac.hl.s0"]
    fn hexagon_M2_mpyu_nac_hl_s0(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpyu.nac.hl.s1"]
    fn hexagon_M2_mpyu_nac_hl_s1(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpyu.nac.lh.s0"]
    fn hexagon_M2_mpyu_nac_lh_s0(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpyu.nac.lh.s1"]
    fn hexagon_M2_mpyu_nac_lh_s1(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpyu.nac.ll.s0"]
    fn hexagon_M2_mpyu_nac_ll_s0(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpyu.nac.ll.s1"]
    fn hexagon_M2_mpyu_nac_ll_s1(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpyu.up"]
    fn hexagon_M2_mpyu_up(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.mpyud.acc.hh.s0"]
    fn hexagon_M2_mpyud_acc_hh_s0(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyud.acc.hh.s1"]
    fn hexagon_M2_mpyud_acc_hh_s1(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyud.acc.hl.s0"]
    fn hexagon_M2_mpyud_acc_hl_s0(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyud.acc.hl.s1"]
    fn hexagon_M2_mpyud_acc_hl_s1(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyud.acc.lh.s0"]
    fn hexagon_M2_mpyud_acc_lh_s0(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyud.acc.lh.s1"]
    fn hexagon_M2_mpyud_acc_lh_s1(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyud.acc.ll.s0"]
    fn hexagon_M2_mpyud_acc_ll_s0(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyud.acc.ll.s1"]
    fn hexagon_M2_mpyud_acc_ll_s1(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyud.hh.s0"]
    fn hexagon_M2_mpyud_hh_s0(_: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyud.hh.s1"]
    fn hexagon_M2_mpyud_hh_s1(_: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyud.hl.s0"]
    fn hexagon_M2_mpyud_hl_s0(_: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyud.hl.s1"]
    fn hexagon_M2_mpyud_hl_s1(_: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyud.lh.s0"]
    fn hexagon_M2_mpyud_lh_s0(_: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyud.lh.s1"]
    fn hexagon_M2_mpyud_lh_s1(_: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyud.ll.s0"]
    fn hexagon_M2_mpyud_ll_s0(_: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyud.ll.s1"]
    fn hexagon_M2_mpyud_ll_s1(_: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyud.nac.hh.s0"]
    fn hexagon_M2_mpyud_nac_hh_s0(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyud.nac.hh.s1"]
    fn hexagon_M2_mpyud_nac_hh_s1(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyud.nac.hl.s0"]
    fn hexagon_M2_mpyud_nac_hl_s0(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyud.nac.hl.s1"]
    fn hexagon_M2_mpyud_nac_hl_s1(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyud.nac.lh.s0"]
    fn hexagon_M2_mpyud_nac_lh_s0(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyud.nac.lh.s1"]
    fn hexagon_M2_mpyud_nac_lh_s1(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyud.nac.ll.s0"]
    fn hexagon_M2_mpyud_nac_ll_s0(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyud.nac.ll.s1"]
    fn hexagon_M2_mpyud_nac_ll_s1(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.mpyui"]
    fn hexagon_M2_mpyui(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.nacci"]
    fn hexagon_M2_nacci(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.naccii"]
    fn hexagon_M2_naccii(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.subacc"]
    fn hexagon_M2_subacc(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.vabsdiffh"]
    fn hexagon_M2_vabsdiffh(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.vabsdiffw"]
    fn hexagon_M2_vabsdiffw(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.vcmac.s0.sat.i"]
    fn hexagon_M2_vcmac_s0_sat_i(_: i64, _: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.vcmac.s0.sat.r"]
    fn hexagon_M2_vcmac_s0_sat_r(_: i64, _: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.vcmpy.s0.sat.i"]
    fn hexagon_M2_vcmpy_s0_sat_i(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.vcmpy.s0.sat.r"]
    fn hexagon_M2_vcmpy_s0_sat_r(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.vcmpy.s1.sat.i"]
    fn hexagon_M2_vcmpy_s1_sat_i(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.vcmpy.s1.sat.r"]
    fn hexagon_M2_vcmpy_s1_sat_r(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.vdmacs.s0"]
    fn hexagon_M2_vdmacs_s0(_: i64, _: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.vdmacs.s1"]
    fn hexagon_M2_vdmacs_s1(_: i64, _: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.vdmpyrs.s0"]
    fn hexagon_M2_vdmpyrs_s0(_: i64, _: i64) -> i32;
    #[link_name = "llvm.hexagon.M2.vdmpyrs.s1"]
    fn hexagon_M2_vdmpyrs_s1(_: i64, _: i64) -> i32;
    #[link_name = "llvm.hexagon.M2.vdmpys.s0"]
    fn hexagon_M2_vdmpys_s0(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.vdmpys.s1"]
    fn hexagon_M2_vdmpys_s1(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.vmac2"]
    fn hexagon_M2_vmac2(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.vmac2es"]
    fn hexagon_M2_vmac2es(_: i64, _: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.vmac2es.s0"]
    fn hexagon_M2_vmac2es_s0(_: i64, _: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.vmac2es.s1"]
    fn hexagon_M2_vmac2es_s1(_: i64, _: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.vmac2s.s0"]
    fn hexagon_M2_vmac2s_s0(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.vmac2s.s1"]
    fn hexagon_M2_vmac2s_s1(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.vmac2su.s0"]
    fn hexagon_M2_vmac2su_s0(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.vmac2su.s1"]
    fn hexagon_M2_vmac2su_s1(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.vmpy2es.s0"]
    fn hexagon_M2_vmpy2es_s0(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.vmpy2es.s1"]
    fn hexagon_M2_vmpy2es_s1(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.vmpy2s.s0"]
    fn hexagon_M2_vmpy2s_s0(_: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.vmpy2s.s0pack"]
    fn hexagon_M2_vmpy2s_s0pack(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.vmpy2s.s1"]
    fn hexagon_M2_vmpy2s_s1(_: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.vmpy2s.s1pack"]
    fn hexagon_M2_vmpy2s_s1pack(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.vmpy2su.s0"]
    fn hexagon_M2_vmpy2su_s0(_: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.vmpy2su.s1"]
    fn hexagon_M2_vmpy2su_s1(_: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.vraddh"]
    fn hexagon_M2_vraddh(_: i64, _: i64) -> i32;
    #[link_name = "llvm.hexagon.M2.vradduh"]
    fn hexagon_M2_vradduh(_: i64, _: i64) -> i32;
    #[link_name = "llvm.hexagon.M2.vrcmaci.s0"]
    fn hexagon_M2_vrcmaci_s0(_: i64, _: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.vrcmaci.s0c"]
    fn hexagon_M2_vrcmaci_s0c(_: i64, _: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.vrcmacr.s0"]
    fn hexagon_M2_vrcmacr_s0(_: i64, _: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.vrcmacr.s0c"]
    fn hexagon_M2_vrcmacr_s0c(_: i64, _: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.vrcmpyi.s0"]
    fn hexagon_M2_vrcmpyi_s0(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.vrcmpyi.s0c"]
    fn hexagon_M2_vrcmpyi_s0c(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.vrcmpyr.s0"]
    fn hexagon_M2_vrcmpyr_s0(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.vrcmpyr.s0c"]
    fn hexagon_M2_vrcmpyr_s0c(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.vrcmpys.acc.s1"]
    fn hexagon_M2_vrcmpys_acc_s1(_: i64, _: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.vrcmpys.s1"]
    fn hexagon_M2_vrcmpys_s1(_: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M2.vrcmpys.s1rp"]
    fn hexagon_M2_vrcmpys_s1rp(_: i64, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M2.vrmac.s0"]
    fn hexagon_M2_vrmac_s0(_: i64, _: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.vrmpy.s0"]
    fn hexagon_M2_vrmpy_s0(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M2.xor.xacc"]
    fn hexagon_M2_xor_xacc(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M4.and.and"]
    fn hexagon_M4_and_and(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M4.and.andn"]
    fn hexagon_M4_and_andn(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M4.and.or"]
    fn hexagon_M4_and_or(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M4.and.xor"]
    fn hexagon_M4_and_xor(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M4.cmpyi.wh"]
    fn hexagon_M4_cmpyi_wh(_: i64, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M4.cmpyi.whc"]
    fn hexagon_M4_cmpyi_whc(_: i64, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M4.cmpyr.wh"]
    fn hexagon_M4_cmpyr_wh(_: i64, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M4.cmpyr.whc"]
    fn hexagon_M4_cmpyr_whc(_: i64, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M4.mac.up.s1.sat"]
    fn hexagon_M4_mac_up_s1_sat(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M4.mpyri.addi"]
    fn hexagon_M4_mpyri_addi(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M4.mpyri.addr"]
    fn hexagon_M4_mpyri_addr(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M4.mpyri.addr.u2"]
    fn hexagon_M4_mpyri_addr_u2(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M4.mpyrr.addi"]
    fn hexagon_M4_mpyrr_addi(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M4.mpyrr.addr"]
    fn hexagon_M4_mpyrr_addr(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M4.nac.up.s1.sat"]
    fn hexagon_M4_nac_up_s1_sat(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M4.or.and"]
    fn hexagon_M4_or_and(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M4.or.andn"]
    fn hexagon_M4_or_andn(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M4.or.or"]
    fn hexagon_M4_or_or(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M4.or.xor"]
    fn hexagon_M4_or_xor(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M4.pmpyw"]
    fn hexagon_M4_pmpyw(_: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M4.pmpyw.acc"]
    fn hexagon_M4_pmpyw_acc(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M4.vpmpyh"]
    fn hexagon_M4_vpmpyh(_: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M4.vpmpyh.acc"]
    fn hexagon_M4_vpmpyh_acc(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M4.vrmpyeh.acc.s0"]
    fn hexagon_M4_vrmpyeh_acc_s0(_: i64, _: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M4.vrmpyeh.acc.s1"]
    fn hexagon_M4_vrmpyeh_acc_s1(_: i64, _: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M4.vrmpyeh.s0"]
    fn hexagon_M4_vrmpyeh_s0(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M4.vrmpyeh.s1"]
    fn hexagon_M4_vrmpyeh_s1(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M4.vrmpyoh.acc.s0"]
    fn hexagon_M4_vrmpyoh_acc_s0(_: i64, _: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M4.vrmpyoh.acc.s1"]
    fn hexagon_M4_vrmpyoh_acc_s1(_: i64, _: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M4.vrmpyoh.s0"]
    fn hexagon_M4_vrmpyoh_s0(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M4.vrmpyoh.s1"]
    fn hexagon_M4_vrmpyoh_s1(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M4.xor.and"]
    fn hexagon_M4_xor_and(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M4.xor.andn"]
    fn hexagon_M4_xor_andn(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M4.xor.or"]
    fn hexagon_M4_xor_or(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M4.xor.xacc"]
    fn hexagon_M4_xor_xacc(_: i64, _: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M5.vdmacbsu"]
    fn hexagon_M5_vdmacbsu(_: i64, _: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M5.vdmpybsu"]
    fn hexagon_M5_vdmpybsu(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M5.vmacbsu"]
    fn hexagon_M5_vmacbsu(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M5.vmacbuu"]
    fn hexagon_M5_vmacbuu(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M5.vmpybsu"]
    fn hexagon_M5_vmpybsu(_: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M5.vmpybuu"]
    fn hexagon_M5_vmpybuu(_: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.M5.vrmacbsu"]
    fn hexagon_M5_vrmacbsu(_: i64, _: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M5.vrmacbuu"]
    fn hexagon_M5_vrmacbuu(_: i64, _: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M5.vrmpybsu"]
    fn hexagon_M5_vrmpybsu(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M5.vrmpybuu"]
    fn hexagon_M5_vrmpybuu(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.S2.addasl.rrri"]
    fn hexagon_S2_addasl_rrri(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.asl.i.p"]
    fn hexagon_S2_asl_i_p(_: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.asl.i.p.acc"]
    fn hexagon_S2_asl_i_p_acc(_: i64, _: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.asl.i.p.and"]
    fn hexagon_S2_asl_i_p_and(_: i64, _: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.asl.i.p.nac"]
    fn hexagon_S2_asl_i_p_nac(_: i64, _: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.asl.i.p.or"]
    fn hexagon_S2_asl_i_p_or(_: i64, _: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.asl.i.p.xacc"]
    fn hexagon_S2_asl_i_p_xacc(_: i64, _: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.asl.i.r"]
    fn hexagon_S2_asl_i_r(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.asl.i.r.acc"]
    fn hexagon_S2_asl_i_r_acc(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.asl.i.r.and"]
    fn hexagon_S2_asl_i_r_and(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.asl.i.r.nac"]
    fn hexagon_S2_asl_i_r_nac(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.asl.i.r.or"]
    fn hexagon_S2_asl_i_r_or(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.asl.i.r.sat"]
    fn hexagon_S2_asl_i_r_sat(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.asl.i.r.xacc"]
    fn hexagon_S2_asl_i_r_xacc(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.asl.i.vh"]
    fn hexagon_S2_asl_i_vh(_: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.asl.i.vw"]
    fn hexagon_S2_asl_i_vw(_: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.asl.r.p"]
    fn hexagon_S2_asl_r_p(_: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.asl.r.p.acc"]
    fn hexagon_S2_asl_r_p_acc(_: i64, _: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.asl.r.p.and"]
    fn hexagon_S2_asl_r_p_and(_: i64, _: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.asl.r.p.nac"]
    fn hexagon_S2_asl_r_p_nac(_: i64, _: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.asl.r.p.or"]
    fn hexagon_S2_asl_r_p_or(_: i64, _: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.asl.r.p.xor"]
    fn hexagon_S2_asl_r_p_xor(_: i64, _: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.asl.r.r"]
    fn hexagon_S2_asl_r_r(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.asl.r.r.acc"]
    fn hexagon_S2_asl_r_r_acc(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.asl.r.r.and"]
    fn hexagon_S2_asl_r_r_and(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.asl.r.r.nac"]
    fn hexagon_S2_asl_r_r_nac(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.asl.r.r.or"]
    fn hexagon_S2_asl_r_r_or(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.asl.r.r.sat"]
    fn hexagon_S2_asl_r_r_sat(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.asl.r.vh"]
    fn hexagon_S2_asl_r_vh(_: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.asl.r.vw"]
    fn hexagon_S2_asl_r_vw(_: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.asr.i.p"]
    fn hexagon_S2_asr_i_p(_: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.asr.i.p.acc"]
    fn hexagon_S2_asr_i_p_acc(_: i64, _: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.asr.i.p.and"]
    fn hexagon_S2_asr_i_p_and(_: i64, _: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.asr.i.p.nac"]
    fn hexagon_S2_asr_i_p_nac(_: i64, _: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.asr.i.p.or"]
    fn hexagon_S2_asr_i_p_or(_: i64, _: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.asr.i.p.rnd"]
    fn hexagon_S2_asr_i_p_rnd(_: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.asr.i.p.rnd.goodsyntax"]
    fn hexagon_S2_asr_i_p_rnd_goodsyntax(_: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.asr.i.r"]
    fn hexagon_S2_asr_i_r(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.asr.i.r.acc"]
    fn hexagon_S2_asr_i_r_acc(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.asr.i.r.and"]
    fn hexagon_S2_asr_i_r_and(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.asr.i.r.nac"]
    fn hexagon_S2_asr_i_r_nac(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.asr.i.r.or"]
    fn hexagon_S2_asr_i_r_or(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.asr.i.r.rnd"]
    fn hexagon_S2_asr_i_r_rnd(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.asr.i.r.rnd.goodsyntax"]
    fn hexagon_S2_asr_i_r_rnd_goodsyntax(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.asr.i.svw.trun"]
    fn hexagon_S2_asr_i_svw_trun(_: i64, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.asr.i.vh"]
    fn hexagon_S2_asr_i_vh(_: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.asr.i.vw"]
    fn hexagon_S2_asr_i_vw(_: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.asr.r.p"]
    fn hexagon_S2_asr_r_p(_: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.asr.r.p.acc"]
    fn hexagon_S2_asr_r_p_acc(_: i64, _: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.asr.r.p.and"]
    fn hexagon_S2_asr_r_p_and(_: i64, _: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.asr.r.p.nac"]
    fn hexagon_S2_asr_r_p_nac(_: i64, _: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.asr.r.p.or"]
    fn hexagon_S2_asr_r_p_or(_: i64, _: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.asr.r.p.xor"]
    fn hexagon_S2_asr_r_p_xor(_: i64, _: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.asr.r.r"]
    fn hexagon_S2_asr_r_r(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.asr.r.r.acc"]
    fn hexagon_S2_asr_r_r_acc(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.asr.r.r.and"]
    fn hexagon_S2_asr_r_r_and(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.asr.r.r.nac"]
    fn hexagon_S2_asr_r_r_nac(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.asr.r.r.or"]
    fn hexagon_S2_asr_r_r_or(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.asr.r.r.sat"]
    fn hexagon_S2_asr_r_r_sat(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.asr.r.svw.trun"]
    fn hexagon_S2_asr_r_svw_trun(_: i64, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.asr.r.vh"]
    fn hexagon_S2_asr_r_vh(_: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.asr.r.vw"]
    fn hexagon_S2_asr_r_vw(_: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.brev"]
    fn hexagon_S2_brev(_: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.brevp"]
    fn hexagon_S2_brevp(_: i64) -> i64;
    #[link_name = "llvm.hexagon.S2.cl0"]
    fn hexagon_S2_cl0(_: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.cl0p"]
    fn hexagon_S2_cl0p(_: i64) -> i32;
    #[link_name = "llvm.hexagon.S2.cl1"]
    fn hexagon_S2_cl1(_: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.cl1p"]
    fn hexagon_S2_cl1p(_: i64) -> i32;
    #[link_name = "llvm.hexagon.S2.clb"]
    fn hexagon_S2_clb(_: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.clbnorm"]
    fn hexagon_S2_clbnorm(_: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.clbp"]
    fn hexagon_S2_clbp(_: i64) -> i32;
    #[link_name = "llvm.hexagon.S2.clrbit.i"]
    fn hexagon_S2_clrbit_i(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.clrbit.r"]
    fn hexagon_S2_clrbit_r(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.ct0"]
    fn hexagon_S2_ct0(_: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.ct0p"]
    fn hexagon_S2_ct0p(_: i64) -> i32;
    #[link_name = "llvm.hexagon.S2.ct1"]
    fn hexagon_S2_ct1(_: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.ct1p"]
    fn hexagon_S2_ct1p(_: i64) -> i32;
    #[link_name = "llvm.hexagon.S2.deinterleave"]
    fn hexagon_S2_deinterleave(_: i64) -> i64;
    #[link_name = "llvm.hexagon.S2.extractu"]
    fn hexagon_S2_extractu(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.extractu.rp"]
    fn hexagon_S2_extractu_rp(_: i32, _: i64) -> i32;
    #[link_name = "llvm.hexagon.S2.extractup"]
    fn hexagon_S2_extractup(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.extractup.rp"]
    fn hexagon_S2_extractup_rp(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.S2.insert"]
    fn hexagon_S2_insert(_: i32, _: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.insert.rp"]
    fn hexagon_S2_insert_rp(_: i32, _: i32, _: i64) -> i32;
    #[link_name = "llvm.hexagon.S2.insertp"]
    fn hexagon_S2_insertp(_: i64, _: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.insertp.rp"]
    fn hexagon_S2_insertp_rp(_: i64, _: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.S2.interleave"]
    fn hexagon_S2_interleave(_: i64) -> i64;
    #[link_name = "llvm.hexagon.S2.lfsp"]
    fn hexagon_S2_lfsp(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.S2.lsl.r.p"]
    fn hexagon_S2_lsl_r_p(_: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.lsl.r.p.acc"]
    fn hexagon_S2_lsl_r_p_acc(_: i64, _: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.lsl.r.p.and"]
    fn hexagon_S2_lsl_r_p_and(_: i64, _: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.lsl.r.p.nac"]
    fn hexagon_S2_lsl_r_p_nac(_: i64, _: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.lsl.r.p.or"]
    fn hexagon_S2_lsl_r_p_or(_: i64, _: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.lsl.r.p.xor"]
    fn hexagon_S2_lsl_r_p_xor(_: i64, _: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.lsl.r.r"]
    fn hexagon_S2_lsl_r_r(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.lsl.r.r.acc"]
    fn hexagon_S2_lsl_r_r_acc(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.lsl.r.r.and"]
    fn hexagon_S2_lsl_r_r_and(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.lsl.r.r.nac"]
    fn hexagon_S2_lsl_r_r_nac(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.lsl.r.r.or"]
    fn hexagon_S2_lsl_r_r_or(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.lsl.r.vh"]
    fn hexagon_S2_lsl_r_vh(_: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.lsl.r.vw"]
    fn hexagon_S2_lsl_r_vw(_: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.lsr.i.p"]
    fn hexagon_S2_lsr_i_p(_: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.lsr.i.p.acc"]
    fn hexagon_S2_lsr_i_p_acc(_: i64, _: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.lsr.i.p.and"]
    fn hexagon_S2_lsr_i_p_and(_: i64, _: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.lsr.i.p.nac"]
    fn hexagon_S2_lsr_i_p_nac(_: i64, _: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.lsr.i.p.or"]
    fn hexagon_S2_lsr_i_p_or(_: i64, _: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.lsr.i.p.xacc"]
    fn hexagon_S2_lsr_i_p_xacc(_: i64, _: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.lsr.i.r"]
    fn hexagon_S2_lsr_i_r(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.lsr.i.r.acc"]
    fn hexagon_S2_lsr_i_r_acc(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.lsr.i.r.and"]
    fn hexagon_S2_lsr_i_r_and(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.lsr.i.r.nac"]
    fn hexagon_S2_lsr_i_r_nac(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.lsr.i.r.or"]
    fn hexagon_S2_lsr_i_r_or(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.lsr.i.r.xacc"]
    fn hexagon_S2_lsr_i_r_xacc(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.lsr.i.vh"]
    fn hexagon_S2_lsr_i_vh(_: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.lsr.i.vw"]
    fn hexagon_S2_lsr_i_vw(_: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.lsr.r.p"]
    fn hexagon_S2_lsr_r_p(_: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.lsr.r.p.acc"]
    fn hexagon_S2_lsr_r_p_acc(_: i64, _: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.lsr.r.p.and"]
    fn hexagon_S2_lsr_r_p_and(_: i64, _: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.lsr.r.p.nac"]
    fn hexagon_S2_lsr_r_p_nac(_: i64, _: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.lsr.r.p.or"]
    fn hexagon_S2_lsr_r_p_or(_: i64, _: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.lsr.r.p.xor"]
    fn hexagon_S2_lsr_r_p_xor(_: i64, _: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.lsr.r.r"]
    fn hexagon_S2_lsr_r_r(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.lsr.r.r.acc"]
    fn hexagon_S2_lsr_r_r_acc(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.lsr.r.r.and"]
    fn hexagon_S2_lsr_r_r_and(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.lsr.r.r.nac"]
    fn hexagon_S2_lsr_r_r_nac(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.lsr.r.r.or"]
    fn hexagon_S2_lsr_r_r_or(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.lsr.r.vh"]
    fn hexagon_S2_lsr_r_vh(_: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.lsr.r.vw"]
    fn hexagon_S2_lsr_r_vw(_: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.packhl"]
    fn hexagon_S2_packhl(_: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.parityp"]
    fn hexagon_S2_parityp(_: i64, _: i64) -> i32;
    #[link_name = "llvm.hexagon.S2.setbit.i"]
    fn hexagon_S2_setbit_i(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.setbit.r"]
    fn hexagon_S2_setbit_r(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.shuffeb"]
    fn hexagon_S2_shuffeb(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.S2.shuffeh"]
    fn hexagon_S2_shuffeh(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.S2.shuffob"]
    fn hexagon_S2_shuffob(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.S2.shuffoh"]
    fn hexagon_S2_shuffoh(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.S2.svsathb"]
    fn hexagon_S2_svsathb(_: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.svsathub"]
    fn hexagon_S2_svsathub(_: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.tableidxb.goodsyntax"]
    fn hexagon_S2_tableidxb_goodsyntax(_: i32, _: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.tableidxd.goodsyntax"]
    fn hexagon_S2_tableidxd_goodsyntax(_: i32, _: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.tableidxh.goodsyntax"]
    fn hexagon_S2_tableidxh_goodsyntax(_: i32, _: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.tableidxw.goodsyntax"]
    fn hexagon_S2_tableidxw_goodsyntax(_: i32, _: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.togglebit.i"]
    fn hexagon_S2_togglebit_i(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.togglebit.r"]
    fn hexagon_S2_togglebit_r(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.tstbit.i"]
    fn hexagon_S2_tstbit_i(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.tstbit.r"]
    fn hexagon_S2_tstbit_r(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.valignib"]
    fn hexagon_S2_valignib(_: i64, _: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.valignrb"]
    fn hexagon_S2_valignrb(_: i64, _: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.vcnegh"]
    fn hexagon_S2_vcnegh(_: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.vcrotate"]
    fn hexagon_S2_vcrotate(_: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.vrcnegh"]
    fn hexagon_S2_vrcnegh(_: i64, _: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.vrndpackwh"]
    fn hexagon_S2_vrndpackwh(_: i64) -> i32;
    #[link_name = "llvm.hexagon.S2.vrndpackwhs"]
    fn hexagon_S2_vrndpackwhs(_: i64) -> i32;
    #[link_name = "llvm.hexagon.S2.vsathb"]
    fn hexagon_S2_vsathb(_: i64) -> i32;
    #[link_name = "llvm.hexagon.S2.vsathb.nopack"]
    fn hexagon_S2_vsathb_nopack(_: i64) -> i64;
    #[link_name = "llvm.hexagon.S2.vsathub"]
    fn hexagon_S2_vsathub(_: i64) -> i32;
    #[link_name = "llvm.hexagon.S2.vsathub.nopack"]
    fn hexagon_S2_vsathub_nopack(_: i64) -> i64;
    #[link_name = "llvm.hexagon.S2.vsatwh"]
    fn hexagon_S2_vsatwh(_: i64) -> i32;
    #[link_name = "llvm.hexagon.S2.vsatwh.nopack"]
    fn hexagon_S2_vsatwh_nopack(_: i64) -> i64;
    #[link_name = "llvm.hexagon.S2.vsatwuh"]
    fn hexagon_S2_vsatwuh(_: i64) -> i32;
    #[link_name = "llvm.hexagon.S2.vsatwuh.nopack"]
    fn hexagon_S2_vsatwuh_nopack(_: i64) -> i64;
    #[link_name = "llvm.hexagon.S2.vsplatrb"]
    fn hexagon_S2_vsplatrb(_: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.vsplatrh"]
    fn hexagon_S2_vsplatrh(_: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.vspliceib"]
    fn hexagon_S2_vspliceib(_: i64, _: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.vsplicerb"]
    fn hexagon_S2_vsplicerb(_: i64, _: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.vsxtbh"]
    fn hexagon_S2_vsxtbh(_: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.vsxthw"]
    fn hexagon_S2_vsxthw(_: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.vtrunehb"]
    fn hexagon_S2_vtrunehb(_: i64) -> i32;
    #[link_name = "llvm.hexagon.S2.vtrunewh"]
    fn hexagon_S2_vtrunewh(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.S2.vtrunohb"]
    fn hexagon_S2_vtrunohb(_: i64) -> i32;
    #[link_name = "llvm.hexagon.S2.vtrunowh"]
    fn hexagon_S2_vtrunowh(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.S2.vzxtbh"]
    fn hexagon_S2_vzxtbh(_: i32) -> i64;
    #[link_name = "llvm.hexagon.S2.vzxthw"]
    fn hexagon_S2_vzxthw(_: i32) -> i64;
    #[link_name = "llvm.hexagon.S4.addaddi"]
    fn hexagon_S4_addaddi(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S4.addi.asl.ri"]
    fn hexagon_S4_addi_asl_ri(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S4.addi.lsr.ri"]
    fn hexagon_S4_addi_lsr_ri(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S4.andi.asl.ri"]
    fn hexagon_S4_andi_asl_ri(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S4.andi.lsr.ri"]
    fn hexagon_S4_andi_lsr_ri(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S4.clbaddi"]
    fn hexagon_S4_clbaddi(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S4.clbpaddi"]
    fn hexagon_S4_clbpaddi(_: i64, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S4.clbpnorm"]
    fn hexagon_S4_clbpnorm(_: i64) -> i32;
    #[link_name = "llvm.hexagon.S4.extract"]
    fn hexagon_S4_extract(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S4.extract.rp"]
    fn hexagon_S4_extract_rp(_: i32, _: i64) -> i32;
    #[link_name = "llvm.hexagon.S4.extractp"]
    fn hexagon_S4_extractp(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S4.extractp.rp"]
    fn hexagon_S4_extractp_rp(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.S4.lsli"]
    fn hexagon_S4_lsli(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S4.ntstbit.i"]
    fn hexagon_S4_ntstbit_i(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S4.ntstbit.r"]
    fn hexagon_S4_ntstbit_r(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S4.or.andi"]
    fn hexagon_S4_or_andi(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S4.or.andix"]
    fn hexagon_S4_or_andix(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S4.or.ori"]
    fn hexagon_S4_or_ori(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S4.ori.asl.ri"]
    fn hexagon_S4_ori_asl_ri(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S4.ori.lsr.ri"]
    fn hexagon_S4_ori_lsr_ri(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S4.parity"]
    fn hexagon_S4_parity(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S4.subaddi"]
    fn hexagon_S4_subaddi(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S4.subi.asl.ri"]
    fn hexagon_S4_subi_asl_ri(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S4.subi.lsr.ri"]
    fn hexagon_S4_subi_lsr_ri(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S4.vrcrotate"]
    fn hexagon_S4_vrcrotate(_: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S4.vrcrotate.acc"]
    fn hexagon_S4_vrcrotate_acc(_: i64, _: i64, _: i32, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S4.vxaddsubh"]
    fn hexagon_S4_vxaddsubh(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.S4.vxaddsubhr"]
    fn hexagon_S4_vxaddsubhr(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.S4.vxaddsubw"]
    fn hexagon_S4_vxaddsubw(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.S4.vxsubaddh"]
    fn hexagon_S4_vxsubaddh(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.S4.vxsubaddhr"]
    fn hexagon_S4_vxsubaddhr(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.S4.vxsubaddw"]
    fn hexagon_S4_vxsubaddw(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.S5.asrhub.rnd.sat.goodsyntax"]
    fn hexagon_S5_asrhub_rnd_sat_goodsyntax(_: i64, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S5.asrhub.sat"]
    fn hexagon_S5_asrhub_sat(_: i64, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S5.popcountp"]
    fn hexagon_S5_popcountp(_: i64) -> i32;
    #[link_name = "llvm.hexagon.S5.vasrhrnd.goodsyntax"]
    fn hexagon_S5_vasrhrnd_goodsyntax(_: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.Y2.dccleana"]
    fn hexagon_Y2_dccleana(_: i32);
    #[link_name = "llvm.hexagon.Y2.dccleaninva"]
    fn hexagon_Y2_dccleaninva(_: i32);
    #[link_name = "llvm.hexagon.Y2.dcfetch"]
    fn hexagon_Y2_dcfetch(_: i32);
    #[link_name = "llvm.hexagon.Y2.dcinva"]
    fn hexagon_Y2_dcinva(_: i32);
    #[link_name = "llvm.hexagon.Y2.dczeroa"]
    fn hexagon_Y2_dczeroa(_: i32);
    #[link_name = "llvm.hexagon.Y4.l2fetch"]
    fn hexagon_Y4_l2fetch(_: i32, _: i32);
    #[link_name = "llvm.hexagon.Y5.l2fetch"]
    fn hexagon_Y5_l2fetch(_: i32, _: i64);
    #[link_name = "llvm.hexagon.S6.rol.i.p"]
    fn hexagon_S6_rol_i_p(_: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S6.rol.i.p.acc"]
    fn hexagon_S6_rol_i_p_acc(_: i64, _: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S6.rol.i.p.and"]
    fn hexagon_S6_rol_i_p_and(_: i64, _: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S6.rol.i.p.nac"]
    fn hexagon_S6_rol_i_p_nac(_: i64, _: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S6.rol.i.p.or"]
    fn hexagon_S6_rol_i_p_or(_: i64, _: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S6.rol.i.p.xacc"]
    fn hexagon_S6_rol_i_p_xacc(_: i64, _: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.S6.rol.i.r"]
    fn hexagon_S6_rol_i_r(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S6.rol.i.r.acc"]
    fn hexagon_S6_rol_i_r_acc(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S6.rol.i.r.and"]
    fn hexagon_S6_rol_i_r_and(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S6.rol.i.r.nac"]
    fn hexagon_S6_rol_i_r_nac(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S6.rol.i.r.or"]
    fn hexagon_S6_rol_i_r_or(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S6.rol.i.r.xacc"]
    fn hexagon_S6_rol_i_r_xacc(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.M6.vabsdiffb"]
    fn hexagon_M6_vabsdiffb(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M6.vabsdiffub"]
    fn hexagon_M6_vabsdiffub(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.S6.vsplatrbp"]
    fn hexagon_S6_vsplatrbp(_: i32) -> i64;
    #[link_name = "llvm.hexagon.S6.vtrunehb.ppp"]
    fn hexagon_S6_vtrunehb_ppp(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.S6.vtrunohb.ppp"]
    fn hexagon_S6_vtrunohb_ppp(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.A6.vcmpbeq.notany"]
    fn hexagon_A6_vcmpbeq_notany(_: i64, _: i64) -> i32;
    #[link_name = "llvm.hexagon.F2.dfadd"]
    fn hexagon_F2_dfadd(_: f64, _: f64) -> f64;
    #[link_name = "llvm.hexagon.F2.dfsub"]
    fn hexagon_F2_dfsub(_: f64, _: f64) -> f64;
    #[link_name = "llvm.hexagon.M2.mnaci"]
    fn hexagon_M2_mnaci(_: i32, _: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.S2.mask"]
    fn hexagon_S2_mask(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A7.clip"]
    fn hexagon_A7_clip(_: i32, _: i32) -> i32;
    #[link_name = "llvm.hexagon.A7.croundd.ri"]
    fn hexagon_A7_croundd_ri(_: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.A7.croundd.rr"]
    fn hexagon_A7_croundd_rr(_: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.A7.vclip"]
    fn hexagon_A7_vclip(_: i64, _: i32) -> i64;
    #[link_name = "llvm.hexagon.F2.dfmax"]
    fn hexagon_F2_dfmax(_: f64, _: f64) -> f64;
    #[link_name = "llvm.hexagon.F2.dfmin"]
    fn hexagon_F2_dfmin(_: f64, _: f64) -> f64;
    #[link_name = "llvm.hexagon.F2.dfmpyfix"]
    fn hexagon_F2_dfmpyfix(_: f64, _: f64) -> f64;
    #[link_name = "llvm.hexagon.F2.dfmpyhh"]
    fn hexagon_F2_dfmpyhh(_: f64, _: f64, _: f64) -> f64;
    #[link_name = "llvm.hexagon.F2.dfmpylh"]
    fn hexagon_F2_dfmpylh(_: f64, _: f64, _: f64) -> f64;
    #[link_name = "llvm.hexagon.F2.dfmpyll"]
    fn hexagon_F2_dfmpyll(_: f64, _: f64) -> f64;
    #[link_name = "llvm.hexagon.M7.dcmpyiw"]
    fn hexagon_M7_dcmpyiw(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M7.dcmpyiw.acc"]
    fn hexagon_M7_dcmpyiw_acc(_: i64, _: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M7.dcmpyiwc"]
    fn hexagon_M7_dcmpyiwc(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M7.dcmpyiwc.acc"]
    fn hexagon_M7_dcmpyiwc_acc(_: i64, _: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M7.dcmpyrw"]
    fn hexagon_M7_dcmpyrw(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M7.dcmpyrw.acc"]
    fn hexagon_M7_dcmpyrw_acc(_: i64, _: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M7.dcmpyrwc"]
    fn hexagon_M7_dcmpyrwc(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M7.dcmpyrwc.acc"]
    fn hexagon_M7_dcmpyrwc_acc(_: i64, _: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M7.vdmpy"]
    fn hexagon_M7_vdmpy(_: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M7.vdmpy.acc"]
    fn hexagon_M7_vdmpy_acc(_: i64, _: i64, _: i64) -> i64;
    #[link_name = "llvm.hexagon.M7.wcmpyiw"]
    fn hexagon_M7_wcmpyiw(_: i64, _: i64) -> i32;
    #[link_name = "llvm.hexagon.M7.wcmpyiw.rnd"]
    fn hexagon_M7_wcmpyiw_rnd(_: i64, _: i64) -> i32;
    #[link_name = "llvm.hexagon.M7.wcmpyiwc"]
    fn hexagon_M7_wcmpyiwc(_: i64, _: i64) -> i32;
    #[link_name = "llvm.hexagon.M7.wcmpyiwc.rnd"]
    fn hexagon_M7_wcmpyiwc_rnd(_: i64, _: i64) -> i32;
    #[link_name = "llvm.hexagon.M7.wcmpyrw"]
    fn hexagon_M7_wcmpyrw(_: i64, _: i64) -> i32;
    #[link_name = "llvm.hexagon.M7.wcmpyrw.rnd"]
    fn hexagon_M7_wcmpyrw_rnd(_: i64, _: i64) -> i32;
    #[link_name = "llvm.hexagon.M7.wcmpyrwc"]
    fn hexagon_M7_wcmpyrwc(_: i64, _: i64) -> i32;
    #[link_name = "llvm.hexagon.M7.wcmpyrwc.rnd"]
    fn hexagon_M7_wcmpyrwc_rnd(_: i64, _: i64) -> i32;
    #[link_name = "llvm.hexagon.Y6.dmlink"]
    fn hexagon_Y6_dmlink(_: i32, _: i32);
    #[link_name = "llvm.hexagon.Y6.dmpause"]
    fn hexagon_Y6_dmpause() -> i32;
    #[link_name = "llvm.hexagon.Y6.dmpoll"]
    fn hexagon_Y6_dmpoll() -> i32;
    #[link_name = "llvm.hexagon.Y6.dmresume"]
    fn hexagon_Y6_dmresume(_: i32);
    #[link_name = "llvm.hexagon.Y6.dmstart"]
    fn hexagon_Y6_dmstart(_: i32);
    #[link_name = "llvm.hexagon.Y6.dmwait"]
    fn hexagon_Y6_dmwait() -> i32;
}

/// `Rd32=abs(Rs32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(abs))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_abs_R(rs: i32) -> i32 {
    hexagon_A2_abs(rs)
}

/// `Rdd32=abs(Rss32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(abs))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_abs_P(rss: i64) -> i64 {
    hexagon_A2_absp(rss)
}

/// `Rd32=abs(Rs32):sat`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(abs))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_abs_R_sat(rs: i32) -> i32 {
    hexagon_A2_abssat(rs)
}

/// `Rd32=add(Rs32,Rt32)`
///
/// Instruction Type: ALU32_3op
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(test, assert_instr(add))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_add_RR(rs: i32, rt: i32) -> i32 {
    hexagon_A2_add(rs, rt)
}

/// `Rd32=add(Rt32.h,Rs32.h):<<16`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(add))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_add_RhRh_s16(rt: i32, rs: i32) -> i32 {
    hexagon_A2_addh_h16_hh(rt, rs)
}

/// `Rd32=add(Rt32.h,Rs32.l):<<16`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(add))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_add_RhRl_s16(rt: i32, rs: i32) -> i32 {
    hexagon_A2_addh_h16_hl(rt, rs)
}

/// `Rd32=add(Rt32.l,Rs32.h):<<16`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(add))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_add_RlRh_s16(rt: i32, rs: i32) -> i32 {
    hexagon_A2_addh_h16_lh(rt, rs)
}

/// `Rd32=add(Rt32.l,Rs32.l):<<16`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(add))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_add_RlRl_s16(rt: i32, rs: i32) -> i32 {
    hexagon_A2_addh_h16_ll(rt, rs)
}

/// `Rd32=add(Rt32.h,Rs32.h):sat:<<16`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(add))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_add_RhRh_sat_s16(rt: i32, rs: i32) -> i32 {
    hexagon_A2_addh_h16_sat_hh(rt, rs)
}

/// `Rd32=add(Rt32.h,Rs32.l):sat:<<16`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(add))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_add_RhRl_sat_s16(rt: i32, rs: i32) -> i32 {
    hexagon_A2_addh_h16_sat_hl(rt, rs)
}

/// `Rd32=add(Rt32.l,Rs32.h):sat:<<16`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(add))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_add_RlRh_sat_s16(rt: i32, rs: i32) -> i32 {
    hexagon_A2_addh_h16_sat_lh(rt, rs)
}

/// `Rd32=add(Rt32.l,Rs32.l):sat:<<16`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(add))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_add_RlRl_sat_s16(rt: i32, rs: i32) -> i32 {
    hexagon_A2_addh_h16_sat_ll(rt, rs)
}

/// `Rd32=add(Rt32.l,Rs32.h)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(add))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_add_RlRh(rt: i32, rs: i32) -> i32 {
    hexagon_A2_addh_l16_hl(rt, rs)
}

/// `Rd32=add(Rt32.l,Rs32.l)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(add))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_add_RlRl(rt: i32, rs: i32) -> i32 {
    hexagon_A2_addh_l16_ll(rt, rs)
}

/// `Rd32=add(Rt32.l,Rs32.h):sat`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(add))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_add_RlRh_sat(rt: i32, rs: i32) -> i32 {
    hexagon_A2_addh_l16_sat_hl(rt, rs)
}

/// `Rd32=add(Rt32.l,Rs32.l):sat`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(add))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_add_RlRl_sat(rt: i32, rs: i32) -> i32 {
    hexagon_A2_addh_l16_sat_ll(rt, rs)
}

/// `Rd32=add(Rs32,#s16)`
///
/// Instruction Type: ALU32_ADDI
/// Execution Slots: SLOT0123
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(add, IS16 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_add_RI<const IS16: i32>(rs: i32) -> i32 {
    static_assert_simm_bits!(IS16, 16);
    hexagon_A2_addi(rs, IS16)
}

/// `Rdd32=add(Rss32,Rtt32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(add))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_add_PP(rss: i64, rtt: i64) -> i64 {
    hexagon_A2_addp(rss, rtt)
}

/// `Rdd32=add(Rss32,Rtt32):sat`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(add))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_add_PP_sat(rss: i64, rtt: i64) -> i64 {
    hexagon_A2_addpsat(rss, rtt)
}

/// `Rd32=add(Rs32,Rt32):sat`
///
/// Instruction Type: ALU32_3op
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(test, assert_instr(add))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_add_RR_sat(rs: i32, rt: i32) -> i32 {
    hexagon_A2_addsat(rs, rt)
}

/// `Rdd32=add(Rs32,Rtt32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(test, assert_instr(add))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_add_RP(rs: i32, rtt: i64) -> i64 {
    hexagon_A2_addsp(rs, rtt)
}

/// `Rd32=and(Rs32,Rt32)`
///
/// Instruction Type: ALU32_3op
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(test, assert_instr(and))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_and_RR(rs: i32, rt: i32) -> i32 {
    hexagon_A2_and(rs, rt)
}

/// `Rd32=and(Rs32,#s10)`
///
/// Instruction Type: ALU32_2op
/// Execution Slots: SLOT0123
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(and, IS10 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_and_RI<const IS10: i32>(rs: i32) -> i32 {
    static_assert_simm_bits!(IS10, 10);
    hexagon_A2_andir(rs, IS10)
}

/// `Rdd32=and(Rss32,Rtt32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(and))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_and_PP(rss: i64, rtt: i64) -> i64 {
    hexagon_A2_andp(rss, rtt)
}

/// `Rd32=aslh(Rs32)`
///
/// Instruction Type: ALU32_2op
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(test, assert_instr(aslh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_aslh_R(rs: i32) -> i32 {
    hexagon_A2_aslh(rs)
}

/// `Rd32=asrh(Rs32)`
///
/// Instruction Type: ALU32_2op
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(test, assert_instr(asrh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_asrh_R(rs: i32) -> i32 {
    hexagon_A2_asrh(rs)
}

/// `Rd32=combine(Rt32.h,Rs32.h)`
///
/// Instruction Type: ALU32_3op
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(test, assert_instr(combine))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_combine_RhRh(rt: i32, rs: i32) -> i32 {
    hexagon_A2_combine_hh(rt, rs)
}

/// `Rd32=combine(Rt32.h,Rs32.l)`
///
/// Instruction Type: ALU32_3op
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(test, assert_instr(combine))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_combine_RhRl(rt: i32, rs: i32) -> i32 {
    hexagon_A2_combine_hl(rt, rs)
}

/// `Rd32=combine(Rt32.l,Rs32.h)`
///
/// Instruction Type: ALU32_3op
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(test, assert_instr(combine))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_combine_RlRh(rt: i32, rs: i32) -> i32 {
    hexagon_A2_combine_lh(rt, rs)
}

/// `Rd32=combine(Rt32.l,Rs32.l)`
///
/// Instruction Type: ALU32_3op
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(test, assert_instr(combine))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_combine_RlRl(rt: i32, rs: i32) -> i32 {
    hexagon_A2_combine_ll(rt, rs)
}

/// `Rdd32=combine(#s8,#S8)`
///
/// Instruction Type: ALU32_2op
/// Execution Slots: SLOT0123
#[inline(always)]
#[rustc_legacy_const_generics(0, 1)]
#[cfg_attr(test, assert_instr(combine, IS8 = 0, IS8_2 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_combine_II<const IS8: i32, const IS8_2: i32>() -> i64 {
    static_assert_simm_bits!(IS8, 8);
    static_assert_simm_bits!(IS8_2, 8);
    hexagon_A2_combineii(IS8, IS8_2)
}

/// `Rdd32=combine(Rs32,Rt32)`
///
/// Instruction Type: ALU32_3op
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(test, assert_instr(combine))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_combine_RR(rs: i32, rt: i32) -> i64 {
    hexagon_A2_combinew(rs, rt)
}

/// `Rd32=max(Rs32,Rt32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(max))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_max_RR(rs: i32, rt: i32) -> i32 {
    hexagon_A2_max(rs, rt)
}

/// `Rdd32=max(Rss32,Rtt32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(max))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_max_PP(rss: i64, rtt: i64) -> i64 {
    hexagon_A2_maxp(rss, rtt)
}

/// `Rd32=maxu(Rs32,Rt32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(maxu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_maxu_RR(rs: i32, rt: i32) -> i32 {
    hexagon_A2_maxu(rs, rt)
}

/// `Rdd32=maxu(Rss32,Rtt32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(maxu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_maxu_PP(rss: i64, rtt: i64) -> i64 {
    hexagon_A2_maxup(rss, rtt)
}

/// `Rd32=min(Rt32,Rs32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(min))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_min_RR(rt: i32, rs: i32) -> i32 {
    hexagon_A2_min(rt, rs)
}

/// `Rdd32=min(Rtt32,Rss32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(min))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_min_PP(rtt: i64, rss: i64) -> i64 {
    hexagon_A2_minp(rtt, rss)
}

/// `Rd32=minu(Rt32,Rs32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(minu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_minu_RR(rt: i32, rs: i32) -> i32 {
    hexagon_A2_minu(rt, rs)
}

/// `Rdd32=minu(Rtt32,Rss32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(minu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_minu_PP(rtt: i64, rss: i64) -> i64 {
    hexagon_A2_minup(rtt, rss)
}

/// `Rd32=neg(Rs32)`
///
/// Instruction Type: ALU32_2op
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(test, assert_instr(neg))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_neg_R(rs: i32) -> i32 {
    hexagon_A2_neg(rs)
}

/// `Rdd32=neg(Rss32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(neg))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_neg_P(rss: i64) -> i64 {
    hexagon_A2_negp(rss)
}

/// `Rd32=neg(Rs32):sat`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(neg))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_neg_R_sat(rs: i32) -> i32 {
    hexagon_A2_negsat(rs)
}

/// `Rd32=not(Rs32)`
///
/// Instruction Type: ALU32_2op
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(test, assert_instr(not))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_not_R(rs: i32) -> i32 {
    hexagon_A2_not(rs)
}

/// `Rdd32=not(Rss32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(not))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_not_P(rss: i64) -> i64 {
    hexagon_A2_notp(rss)
}

/// `Rd32=or(Rs32,Rt32)`
///
/// Instruction Type: ALU32_3op
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(test, assert_instr(or))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_or_RR(rs: i32, rt: i32) -> i32 {
    hexagon_A2_or(rs, rt)
}

/// `Rd32=or(Rs32,#s10)`
///
/// Instruction Type: ALU32_2op
/// Execution Slots: SLOT0123
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(or, IS10 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_or_RI<const IS10: i32>(rs: i32) -> i32 {
    static_assert_simm_bits!(IS10, 10);
    hexagon_A2_orir(rs, IS10)
}

/// `Rdd32=or(Rss32,Rtt32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(or))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_or_PP(rss: i64, rtt: i64) -> i64 {
    hexagon_A2_orp(rss, rtt)
}

/// `Rd32=round(Rss32):sat`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(round))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_round_P_sat(rss: i64) -> i32 {
    hexagon_A2_roundsat(rss)
}

/// `Rd32=sat(Rss32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(sat))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_sat_P(rss: i64) -> i32 {
    hexagon_A2_sat(rss)
}

/// `Rd32=satb(Rs32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(satb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_satb_R(rs: i32) -> i32 {
    hexagon_A2_satb(rs)
}

/// `Rd32=sath(Rs32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(sath))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_sath_R(rs: i32) -> i32 {
    hexagon_A2_sath(rs)
}

/// `Rd32=satub(Rs32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(satub))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_satub_R(rs: i32) -> i32 {
    hexagon_A2_satub(rs)
}

/// `Rd32=satuh(Rs32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(satuh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_satuh_R(rs: i32) -> i32 {
    hexagon_A2_satuh(rs)
}

/// `Rd32=sub(Rt32,Rs32)`
///
/// Instruction Type: ALU32_3op
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(test, assert_instr(sub))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_sub_RR(rt: i32, rs: i32) -> i32 {
    hexagon_A2_sub(rt, rs)
}

/// `Rd32=sub(Rt32.h,Rs32.h):<<16`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(sub))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_sub_RhRh_s16(rt: i32, rs: i32) -> i32 {
    hexagon_A2_subh_h16_hh(rt, rs)
}

/// `Rd32=sub(Rt32.h,Rs32.l):<<16`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(sub))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_sub_RhRl_s16(rt: i32, rs: i32) -> i32 {
    hexagon_A2_subh_h16_hl(rt, rs)
}

/// `Rd32=sub(Rt32.l,Rs32.h):<<16`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(sub))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_sub_RlRh_s16(rt: i32, rs: i32) -> i32 {
    hexagon_A2_subh_h16_lh(rt, rs)
}

/// `Rd32=sub(Rt32.l,Rs32.l):<<16`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(sub))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_sub_RlRl_s16(rt: i32, rs: i32) -> i32 {
    hexagon_A2_subh_h16_ll(rt, rs)
}

/// `Rd32=sub(Rt32.h,Rs32.h):sat:<<16`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(sub))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_sub_RhRh_sat_s16(rt: i32, rs: i32) -> i32 {
    hexagon_A2_subh_h16_sat_hh(rt, rs)
}

/// `Rd32=sub(Rt32.h,Rs32.l):sat:<<16`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(sub))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_sub_RhRl_sat_s16(rt: i32, rs: i32) -> i32 {
    hexagon_A2_subh_h16_sat_hl(rt, rs)
}

/// `Rd32=sub(Rt32.l,Rs32.h):sat:<<16`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(sub))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_sub_RlRh_sat_s16(rt: i32, rs: i32) -> i32 {
    hexagon_A2_subh_h16_sat_lh(rt, rs)
}

/// `Rd32=sub(Rt32.l,Rs32.l):sat:<<16`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(sub))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_sub_RlRl_sat_s16(rt: i32, rs: i32) -> i32 {
    hexagon_A2_subh_h16_sat_ll(rt, rs)
}

/// `Rd32=sub(Rt32.l,Rs32.h)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(sub))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_sub_RlRh(rt: i32, rs: i32) -> i32 {
    hexagon_A2_subh_l16_hl(rt, rs)
}

/// `Rd32=sub(Rt32.l,Rs32.l)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(sub))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_sub_RlRl(rt: i32, rs: i32) -> i32 {
    hexagon_A2_subh_l16_ll(rt, rs)
}

/// `Rd32=sub(Rt32.l,Rs32.h):sat`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(sub))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_sub_RlRh_sat(rt: i32, rs: i32) -> i32 {
    hexagon_A2_subh_l16_sat_hl(rt, rs)
}

/// `Rd32=sub(Rt32.l,Rs32.l):sat`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(sub))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_sub_RlRl_sat(rt: i32, rs: i32) -> i32 {
    hexagon_A2_subh_l16_sat_ll(rt, rs)
}

/// `Rdd32=sub(Rtt32,Rss32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(sub))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_sub_PP(rtt: i64, rss: i64) -> i64 {
    hexagon_A2_subp(rtt, rss)
}

/// `Rd32=sub(#s10,Rs32)`
///
/// Instruction Type: ALU32_2op
/// Execution Slots: SLOT0123
#[inline(always)]
#[rustc_legacy_const_generics(0)]
#[cfg_attr(test, assert_instr(sub, IS10 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_sub_IR<const IS10: i32>(rs: i32) -> i32 {
    static_assert_simm_bits!(IS10, 10);
    hexagon_A2_subri(IS10, rs)
}

/// `Rd32=sub(Rt32,Rs32):sat`
///
/// Instruction Type: ALU32_3op
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(test, assert_instr(sub))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_sub_RR_sat(rt: i32, rs: i32) -> i32 {
    hexagon_A2_subsat(rt, rs)
}

/// `Rd32=vaddh(Rs32,Rt32)`
///
/// Instruction Type: ALU32_3op
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(test, assert_instr(vaddh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_vaddh_RR(rs: i32, rt: i32) -> i32 {
    hexagon_A2_svaddh(rs, rt)
}

/// `Rd32=vaddh(Rs32,Rt32):sat`
///
/// Instruction Type: ALU32_3op
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(test, assert_instr(vaddh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_vaddh_RR_sat(rs: i32, rt: i32) -> i32 {
    hexagon_A2_svaddhs(rs, rt)
}

/// `Rd32=vadduh(Rs32,Rt32):sat`
///
/// Instruction Type: ALU32_3op
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(test, assert_instr(vadduh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_vadduh_RR_sat(rs: i32, rt: i32) -> i32 {
    hexagon_A2_svadduhs(rs, rt)
}

/// `Rd32=vavgh(Rs32,Rt32)`
///
/// Instruction Type: ALU32_3op
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(test, assert_instr(vavgh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_vavgh_RR(rs: i32, rt: i32) -> i32 {
    hexagon_A2_svavgh(rs, rt)
}

/// `Rd32=vavgh(Rs32,Rt32):rnd`
///
/// Instruction Type: ALU32_3op
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(test, assert_instr(vavgh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_vavgh_RR_rnd(rs: i32, rt: i32) -> i32 {
    hexagon_A2_svavghs(rs, rt)
}

/// `Rd32=vnavgh(Rt32,Rs32)`
///
/// Instruction Type: ALU32_3op
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(test, assert_instr(vnavgh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_vnavgh_RR(rt: i32, rs: i32) -> i32 {
    hexagon_A2_svnavgh(rt, rs)
}

/// `Rd32=vsubh(Rt32,Rs32)`
///
/// Instruction Type: ALU32_3op
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(test, assert_instr(vsubh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_vsubh_RR(rt: i32, rs: i32) -> i32 {
    hexagon_A2_svsubh(rt, rs)
}

/// `Rd32=vsubh(Rt32,Rs32):sat`
///
/// Instruction Type: ALU32_3op
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(test, assert_instr(vsubh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_vsubh_RR_sat(rt: i32, rs: i32) -> i32 {
    hexagon_A2_svsubhs(rt, rs)
}

/// `Rd32=vsubuh(Rt32,Rs32):sat`
///
/// Instruction Type: ALU32_3op
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(test, assert_instr(vsubuh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_vsubuh_RR_sat(rt: i32, rs: i32) -> i32 {
    hexagon_A2_svsubuhs(rt, rs)
}

/// `Rd32=swiz(Rs32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(swiz))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_swiz_R(rs: i32) -> i32 {
    hexagon_A2_swiz(rs)
}

/// `Rd32=sxtb(Rs32)`
///
/// Instruction Type: ALU32_2op
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(test, assert_instr(sxtb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_sxtb_R(rs: i32) -> i32 {
    hexagon_A2_sxtb(rs)
}

/// `Rd32=sxth(Rs32)`
///
/// Instruction Type: ALU32_2op
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(test, assert_instr(sxth))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_sxth_R(rs: i32) -> i32 {
    hexagon_A2_sxth(rs)
}

/// `Rdd32=sxtw(Rs32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(sxtw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_sxtw_R(rs: i32) -> i64 {
    hexagon_A2_sxtw(rs)
}

/// `Rd32=Rs32`
///
/// Instruction Type: ALU32_2op
/// Execution Slots: SLOT0123
#[inline(always)]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_equals_R(rs: i32) -> i32 {
    hexagon_A2_tfr(rs)
}

/// `Rx32.h=#u16`
///
/// Instruction Type: ALU32_2op
/// Execution Slots: SLOT0123
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Rh_equals_I<const IU16: u32>(rx: i32) -> i32 {
    static_assert_uimm_bits!(IU16, 16);
    hexagon_A2_tfrih(rx, IU16 as i32)
}

/// `Rx32.l=#u16`
///
/// Instruction Type: ALU32_2op
/// Execution Slots: SLOT0123
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_Rl_equals_I<const IU16: u32>(rx: i32) -> i32 {
    static_assert_uimm_bits!(IU16, 16);
    hexagon_A2_tfril(rx, IU16 as i32)
}

/// `Rdd32=Rss32`
///
/// Instruction Type: ALU32_2op
/// Execution Slots: SLOT0123
#[inline(always)]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_equals_P(rss: i64) -> i64 {
    hexagon_A2_tfrp(rss)
}

/// `Rdd32=#s8`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT0123
#[inline(always)]
#[rustc_legacy_const_generics(0)]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_equals_I<const IS8: i32>() -> i64 {
    static_assert_simm_bits!(IS8, 8);
    hexagon_A2_tfrpi(IS8)
}

/// `Rd32=#s16`
///
/// Instruction Type: ALU32_2op
/// Execution Slots: SLOT0123
#[inline(always)]
#[rustc_legacy_const_generics(0)]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_equals_I<const IS16: i32>() -> i32 {
    static_assert_simm_bits!(IS16, 16);
    hexagon_A2_tfrsi(IS16)
}

/// `Rdd32=vabsh(Rss32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vabsh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vabsh_P(rss: i64) -> i64 {
    hexagon_A2_vabsh(rss)
}

/// `Rdd32=vabsh(Rss32):sat`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vabsh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vabsh_P_sat(rss: i64) -> i64 {
    hexagon_A2_vabshsat(rss)
}

/// `Rdd32=vabsw(Rss32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vabsw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vabsw_P(rss: i64) -> i64 {
    hexagon_A2_vabsw(rss)
}

/// `Rdd32=vabsw(Rss32):sat`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vabsw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vabsw_P_sat(rss: i64) -> i64 {
    hexagon_A2_vabswsat(rss)
}

/// `Rdd32=vaddb(Rss32,Rtt32)`
///
/// Instruction Type: MAPPING
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(test, assert_instr(vaddb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vaddb_PP(rss: i64, rtt: i64) -> i64 {
    hexagon_A2_vaddb_map(rss, rtt)
}

/// `Rdd32=vaddh(Rss32,Rtt32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vaddh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vaddh_PP(rss: i64, rtt: i64) -> i64 {
    hexagon_A2_vaddh(rss, rtt)
}

/// `Rdd32=vaddh(Rss32,Rtt32):sat`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vaddh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vaddh_PP_sat(rss: i64, rtt: i64) -> i64 {
    hexagon_A2_vaddhs(rss, rtt)
}

/// `Rdd32=vaddub(Rss32,Rtt32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vaddub))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vaddub_PP(rss: i64, rtt: i64) -> i64 {
    hexagon_A2_vaddub(rss, rtt)
}

/// `Rdd32=vaddub(Rss32,Rtt32):sat`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vaddub))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vaddub_PP_sat(rss: i64, rtt: i64) -> i64 {
    hexagon_A2_vaddubs(rss, rtt)
}

/// `Rdd32=vadduh(Rss32,Rtt32):sat`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vadduh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vadduh_PP_sat(rss: i64, rtt: i64) -> i64 {
    hexagon_A2_vadduhs(rss, rtt)
}

/// `Rdd32=vaddw(Rss32,Rtt32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vaddw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vaddw_PP(rss: i64, rtt: i64) -> i64 {
    hexagon_A2_vaddw(rss, rtt)
}

/// `Rdd32=vaddw(Rss32,Rtt32):sat`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vaddw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vaddw_PP_sat(rss: i64, rtt: i64) -> i64 {
    hexagon_A2_vaddws(rss, rtt)
}

/// `Rdd32=vavgh(Rss32,Rtt32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vavgh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vavgh_PP(rss: i64, rtt: i64) -> i64 {
    hexagon_A2_vavgh(rss, rtt)
}

/// `Rdd32=vavgh(Rss32,Rtt32):crnd`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vavgh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vavgh_PP_crnd(rss: i64, rtt: i64) -> i64 {
    hexagon_A2_vavghcr(rss, rtt)
}

/// `Rdd32=vavgh(Rss32,Rtt32):rnd`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vavgh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vavgh_PP_rnd(rss: i64, rtt: i64) -> i64 {
    hexagon_A2_vavghr(rss, rtt)
}

/// `Rdd32=vavgub(Rss32,Rtt32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vavgub))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vavgub_PP(rss: i64, rtt: i64) -> i64 {
    hexagon_A2_vavgub(rss, rtt)
}

/// `Rdd32=vavgub(Rss32,Rtt32):rnd`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vavgub))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vavgub_PP_rnd(rss: i64, rtt: i64) -> i64 {
    hexagon_A2_vavgubr(rss, rtt)
}

/// `Rdd32=vavguh(Rss32,Rtt32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vavguh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vavguh_PP(rss: i64, rtt: i64) -> i64 {
    hexagon_A2_vavguh(rss, rtt)
}

/// `Rdd32=vavguh(Rss32,Rtt32):rnd`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vavguh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vavguh_PP_rnd(rss: i64, rtt: i64) -> i64 {
    hexagon_A2_vavguhr(rss, rtt)
}

/// `Rdd32=vavguw(Rss32,Rtt32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vavguw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vavguw_PP(rss: i64, rtt: i64) -> i64 {
    hexagon_A2_vavguw(rss, rtt)
}

/// `Rdd32=vavguw(Rss32,Rtt32):rnd`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vavguw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vavguw_PP_rnd(rss: i64, rtt: i64) -> i64 {
    hexagon_A2_vavguwr(rss, rtt)
}

/// `Rdd32=vavgw(Rss32,Rtt32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vavgw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vavgw_PP(rss: i64, rtt: i64) -> i64 {
    hexagon_A2_vavgw(rss, rtt)
}

/// `Rdd32=vavgw(Rss32,Rtt32):crnd`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vavgw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vavgw_PP_crnd(rss: i64, rtt: i64) -> i64 {
    hexagon_A2_vavgwcr(rss, rtt)
}

/// `Rdd32=vavgw(Rss32,Rtt32):rnd`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vavgw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vavgw_PP_rnd(rss: i64, rtt: i64) -> i64 {
    hexagon_A2_vavgwr(rss, rtt)
}

/// `Pd4=vcmpb.eq(Rss32,Rtt32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vcmpb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_vcmpb_eq_PP(rss: i64, rtt: i64) -> i32 {
    hexagon_A2_vcmpbeq(rss, rtt)
}

/// `Pd4=vcmpb.gtu(Rss32,Rtt32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vcmpb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_vcmpb_gtu_PP(rss: i64, rtt: i64) -> i32 {
    hexagon_A2_vcmpbgtu(rss, rtt)
}

/// `Pd4=vcmph.eq(Rss32,Rtt32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vcmph))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_vcmph_eq_PP(rss: i64, rtt: i64) -> i32 {
    hexagon_A2_vcmpheq(rss, rtt)
}

/// `Pd4=vcmph.gt(Rss32,Rtt32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vcmph))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_vcmph_gt_PP(rss: i64, rtt: i64) -> i32 {
    hexagon_A2_vcmphgt(rss, rtt)
}

/// `Pd4=vcmph.gtu(Rss32,Rtt32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vcmph))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_vcmph_gtu_PP(rss: i64, rtt: i64) -> i32 {
    hexagon_A2_vcmphgtu(rss, rtt)
}

/// `Pd4=vcmpw.eq(Rss32,Rtt32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vcmpw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_vcmpw_eq_PP(rss: i64, rtt: i64) -> i32 {
    hexagon_A2_vcmpweq(rss, rtt)
}

/// `Pd4=vcmpw.gt(Rss32,Rtt32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vcmpw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_vcmpw_gt_PP(rss: i64, rtt: i64) -> i32 {
    hexagon_A2_vcmpwgt(rss, rtt)
}

/// `Pd4=vcmpw.gtu(Rss32,Rtt32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vcmpw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_vcmpw_gtu_PP(rss: i64, rtt: i64) -> i32 {
    hexagon_A2_vcmpwgtu(rss, rtt)
}

/// `Rdd32=vconj(Rss32):sat`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vconj))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vconj_P_sat(rss: i64) -> i64 {
    hexagon_A2_vconj(rss)
}

/// `Rdd32=vmaxb(Rtt32,Rss32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmaxb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmaxb_PP(rtt: i64, rss: i64) -> i64 {
    hexagon_A2_vmaxb(rtt, rss)
}

/// `Rdd32=vmaxh(Rtt32,Rss32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmaxh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmaxh_PP(rtt: i64, rss: i64) -> i64 {
    hexagon_A2_vmaxh(rtt, rss)
}

/// `Rdd32=vmaxub(Rtt32,Rss32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmaxub))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmaxub_PP(rtt: i64, rss: i64) -> i64 {
    hexagon_A2_vmaxub(rtt, rss)
}

/// `Rdd32=vmaxuh(Rtt32,Rss32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmaxuh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmaxuh_PP(rtt: i64, rss: i64) -> i64 {
    hexagon_A2_vmaxuh(rtt, rss)
}

/// `Rdd32=vmaxuw(Rtt32,Rss32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmaxuw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmaxuw_PP(rtt: i64, rss: i64) -> i64 {
    hexagon_A2_vmaxuw(rtt, rss)
}

/// `Rdd32=vmaxw(Rtt32,Rss32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmaxw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmaxw_PP(rtt: i64, rss: i64) -> i64 {
    hexagon_A2_vmaxw(rtt, rss)
}

/// `Rdd32=vminb(Rtt32,Rss32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vminb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vminb_PP(rtt: i64, rss: i64) -> i64 {
    hexagon_A2_vminb(rtt, rss)
}

/// `Rdd32=vminh(Rtt32,Rss32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vminh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vminh_PP(rtt: i64, rss: i64) -> i64 {
    hexagon_A2_vminh(rtt, rss)
}

/// `Rdd32=vminub(Rtt32,Rss32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vminub))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vminub_PP(rtt: i64, rss: i64) -> i64 {
    hexagon_A2_vminub(rtt, rss)
}

/// `Rdd32=vminuh(Rtt32,Rss32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vminuh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vminuh_PP(rtt: i64, rss: i64) -> i64 {
    hexagon_A2_vminuh(rtt, rss)
}

/// `Rdd32=vminuw(Rtt32,Rss32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vminuw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vminuw_PP(rtt: i64, rss: i64) -> i64 {
    hexagon_A2_vminuw(rtt, rss)
}

/// `Rdd32=vminw(Rtt32,Rss32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vminw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vminw_PP(rtt: i64, rss: i64) -> i64 {
    hexagon_A2_vminw(rtt, rss)
}

/// `Rdd32=vnavgh(Rtt32,Rss32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vnavgh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vnavgh_PP(rtt: i64, rss: i64) -> i64 {
    hexagon_A2_vnavgh(rtt, rss)
}

/// `Rdd32=vnavgh(Rtt32,Rss32):crnd:sat`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vnavgh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vnavgh_PP_crnd_sat(rtt: i64, rss: i64) -> i64 {
    hexagon_A2_vnavghcr(rtt, rss)
}

/// `Rdd32=vnavgh(Rtt32,Rss32):rnd:sat`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vnavgh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vnavgh_PP_rnd_sat(rtt: i64, rss: i64) -> i64 {
    hexagon_A2_vnavghr(rtt, rss)
}

/// `Rdd32=vnavgw(Rtt32,Rss32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vnavgw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vnavgw_PP(rtt: i64, rss: i64) -> i64 {
    hexagon_A2_vnavgw(rtt, rss)
}

/// `Rdd32=vnavgw(Rtt32,Rss32):crnd:sat`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vnavgw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vnavgw_PP_crnd_sat(rtt: i64, rss: i64) -> i64 {
    hexagon_A2_vnavgwcr(rtt, rss)
}

/// `Rdd32=vnavgw(Rtt32,Rss32):rnd:sat`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vnavgw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vnavgw_PP_rnd_sat(rtt: i64, rss: i64) -> i64 {
    hexagon_A2_vnavgwr(rtt, rss)
}

/// `Rdd32=vraddub(Rss32,Rtt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vraddub))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vraddub_PP(rss: i64, rtt: i64) -> i64 {
    hexagon_A2_vraddub(rss, rtt)
}

/// `Rxx32+=vraddub(Rss32,Rtt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vraddub))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vraddubacc_PP(rxx: i64, rss: i64, rtt: i64) -> i64 {
    hexagon_A2_vraddub_acc(rxx, rss, rtt)
}

/// `Rdd32=vrsadub(Rss32,Rtt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vrsadub))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vrsadub_PP(rss: i64, rtt: i64) -> i64 {
    hexagon_A2_vrsadub(rss, rtt)
}

/// `Rxx32+=vrsadub(Rss32,Rtt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vrsadub))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vrsadubacc_PP(rxx: i64, rss: i64, rtt: i64) -> i64 {
    hexagon_A2_vrsadub_acc(rxx, rss, rtt)
}

/// `Rdd32=vsubb(Rss32,Rtt32)`
///
/// Instruction Type: MAPPING
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(test, assert_instr(vsubb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vsubb_PP(rss: i64, rtt: i64) -> i64 {
    hexagon_A2_vsubb_map(rss, rtt)
}

/// `Rdd32=vsubh(Rtt32,Rss32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vsubh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vsubh_PP(rtt: i64, rss: i64) -> i64 {
    hexagon_A2_vsubh(rtt, rss)
}

/// `Rdd32=vsubh(Rtt32,Rss32):sat`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vsubh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vsubh_PP_sat(rtt: i64, rss: i64) -> i64 {
    hexagon_A2_vsubhs(rtt, rss)
}

/// `Rdd32=vsubub(Rtt32,Rss32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vsubub))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vsubub_PP(rtt: i64, rss: i64) -> i64 {
    hexagon_A2_vsubub(rtt, rss)
}

/// `Rdd32=vsubub(Rtt32,Rss32):sat`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vsubub))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vsubub_PP_sat(rtt: i64, rss: i64) -> i64 {
    hexagon_A2_vsububs(rtt, rss)
}

/// `Rdd32=vsubuh(Rtt32,Rss32):sat`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vsubuh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vsubuh_PP_sat(rtt: i64, rss: i64) -> i64 {
    hexagon_A2_vsubuhs(rtt, rss)
}

/// `Rdd32=vsubw(Rtt32,Rss32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vsubw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vsubw_PP(rtt: i64, rss: i64) -> i64 {
    hexagon_A2_vsubw(rtt, rss)
}

/// `Rdd32=vsubw(Rtt32,Rss32):sat`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vsubw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vsubw_PP_sat(rtt: i64, rss: i64) -> i64 {
    hexagon_A2_vsubws(rtt, rss)
}

/// `Rd32=xor(Rs32,Rt32)`
///
/// Instruction Type: ALU32_3op
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(test, assert_instr(xor))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_xor_RR(rs: i32, rt: i32) -> i32 {
    hexagon_A2_xor(rs, rt)
}

/// `Rdd32=xor(Rss32,Rtt32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(xor))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_xor_PP(rss: i64, rtt: i64) -> i64 {
    hexagon_A2_xorp(rss, rtt)
}

/// `Rd32=zxtb(Rs32)`
///
/// Instruction Type: ALU32_2op
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(test, assert_instr(zxtb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_zxtb_R(rs: i32) -> i32 {
    hexagon_A2_zxtb(rs)
}

/// `Rd32=zxth(Rs32)`
///
/// Instruction Type: ALU32_2op
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(test, assert_instr(zxth))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_zxth_R(rs: i32) -> i32 {
    hexagon_A2_zxth(rs)
}

/// `Rd32=and(Rt32,~Rs32)`
///
/// Instruction Type: ALU32_3op
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(test, assert_instr(and))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_and_RnR(rt: i32, rs: i32) -> i32 {
    hexagon_A4_andn(rt, rs)
}

/// `Rdd32=and(Rtt32,~Rss32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(and))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_and_PnP(rtt: i64, rss: i64) -> i64 {
    hexagon_A4_andnp(rtt, rss)
}

/// `Rdd32=bitsplit(Rs32,Rt32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(bitsplit))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_bitsplit_RR(rs: i32, rt: i32) -> i64 {
    hexagon_A4_bitsplit(rs, rt)
}

/// `Rdd32=bitsplit(Rs32,#u5)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(bitsplit, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_bitsplit_RI<const IU5: u32>(rs: i32) -> i64 {
    static_assert_uimm_bits!(IU5, 5);
    hexagon_A4_bitspliti(rs, IU5 as i32)
}

/// `Pd4=boundscheck(Rs32,Rtt32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(test, assert_instr(boundscheck))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_boundscheck_RP(rs: i32, rtt: i64) -> i32 {
    hexagon_A4_boundscheck(rs, rtt)
}

/// `Pd4=cmpb.eq(Rs32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(cmpb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_cmpb_eq_RR(rs: i32, rt: i32) -> i32 {
    hexagon_A4_cmpbeq(rs, rt)
}

/// `Pd4=cmpb.eq(Rs32,#u8)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(cmpb, IU8 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_cmpb_eq_RI<const IU8: u32>(rs: i32) -> i32 {
    static_assert_uimm_bits!(IU8, 8);
    hexagon_A4_cmpbeqi(rs, IU8 as i32)
}

/// `Pd4=cmpb.gt(Rs32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(cmpb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_cmpb_gt_RR(rs: i32, rt: i32) -> i32 {
    hexagon_A4_cmpbgt(rs, rt)
}

/// `Pd4=cmpb.gt(Rs32,#s8)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(cmpb, IS8 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_cmpb_gt_RI<const IS8: i32>(rs: i32) -> i32 {
    static_assert_simm_bits!(IS8, 8);
    hexagon_A4_cmpbgti(rs, IS8)
}

/// `Pd4=cmpb.gtu(Rs32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(cmpb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_cmpb_gtu_RR(rs: i32, rt: i32) -> i32 {
    hexagon_A4_cmpbgtu(rs, rt)
}

/// `Pd4=cmpb.gtu(Rs32,#u7)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(cmpb, IU7 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_cmpb_gtu_RI<const IU7: u32>(rs: i32) -> i32 {
    static_assert_uimm_bits!(IU7, 7);
    hexagon_A4_cmpbgtui(rs, IU7 as i32)
}

/// `Pd4=cmph.eq(Rs32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(cmph))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_cmph_eq_RR(rs: i32, rt: i32) -> i32 {
    hexagon_A4_cmpheq(rs, rt)
}

/// `Pd4=cmph.eq(Rs32,#s8)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(cmph, IS8 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_cmph_eq_RI<const IS8: i32>(rs: i32) -> i32 {
    static_assert_simm_bits!(IS8, 8);
    hexagon_A4_cmpheqi(rs, IS8)
}

/// `Pd4=cmph.gt(Rs32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(cmph))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_cmph_gt_RR(rs: i32, rt: i32) -> i32 {
    hexagon_A4_cmphgt(rs, rt)
}

/// `Pd4=cmph.gt(Rs32,#s8)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(cmph, IS8 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_cmph_gt_RI<const IS8: i32>(rs: i32) -> i32 {
    static_assert_simm_bits!(IS8, 8);
    hexagon_A4_cmphgti(rs, IS8)
}

/// `Pd4=cmph.gtu(Rs32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(cmph))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_cmph_gtu_RR(rs: i32, rt: i32) -> i32 {
    hexagon_A4_cmphgtu(rs, rt)
}

/// `Pd4=cmph.gtu(Rs32,#u7)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(cmph, IU7 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_cmph_gtu_RI<const IU7: u32>(rs: i32) -> i32 {
    static_assert_uimm_bits!(IU7, 7);
    hexagon_A4_cmphgtui(rs, IU7 as i32)
}

/// `Rdd32=combine(#s8,Rs32)`
///
/// Instruction Type: ALU32_2op
/// Execution Slots: SLOT0123
#[inline(always)]
#[rustc_legacy_const_generics(0)]
#[cfg_attr(test, assert_instr(combine, IS8 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_combine_IR<const IS8: i32>(rs: i32) -> i64 {
    static_assert_simm_bits!(IS8, 8);
    hexagon_A4_combineir(IS8, rs)
}

/// `Rdd32=combine(Rs32,#s8)`
///
/// Instruction Type: ALU32_2op
/// Execution Slots: SLOT0123
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(combine, IS8 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_combine_RI<const IS8: i32>(rs: i32) -> i64 {
    static_assert_simm_bits!(IS8, 8);
    hexagon_A4_combineri(rs, IS8)
}

/// `Rd32=cround(Rs32,#u5)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(cround, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_cround_RI<const IU5: u32>(rs: i32) -> i32 {
    static_assert_uimm_bits!(IU5, 5);
    hexagon_A4_cround_ri(rs, IU5 as i32)
}

/// `Rd32=cround(Rs32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(cround))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_cround_RR(rs: i32, rt: i32) -> i32 {
    hexagon_A4_cround_rr(rs, rt)
}

/// `Rd32=modwrap(Rs32,Rt32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(modwrap))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_modwrap_RR(rs: i32, rt: i32) -> i32 {
    hexagon_A4_modwrapu(rs, rt)
}

/// `Rd32=or(Rt32,~Rs32)`
///
/// Instruction Type: ALU32_3op
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(test, assert_instr(or))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_or_RnR(rt: i32, rs: i32) -> i32 {
    hexagon_A4_orn(rt, rs)
}

/// `Rdd32=or(Rtt32,~Rss32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(or))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_or_PnP(rtt: i64, rss: i64) -> i64 {
    hexagon_A4_ornp(rtt, rss)
}

/// `Rd32=cmp.eq(Rs32,Rt32)`
///
/// Instruction Type: ALU32_3op
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(test, assert_instr(cmp))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_cmp_eq_RR(rs: i32, rt: i32) -> i32 {
    hexagon_A4_rcmpeq(rs, rt)
}

/// `Rd32=cmp.eq(Rs32,#s8)`
///
/// Instruction Type: ALU32_2op
/// Execution Slots: SLOT0123
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(cmp, IS8 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_cmp_eq_RI<const IS8: i32>(rs: i32) -> i32 {
    static_assert_simm_bits!(IS8, 8);
    hexagon_A4_rcmpeqi(rs, IS8)
}

/// `Rd32=!cmp.eq(Rs32,Rt32)`
///
/// Instruction Type: ALU32_3op
/// Execution Slots: SLOT0123
#[inline(always)]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_not_cmp_eq_RR(rs: i32, rt: i32) -> i32 {
    hexagon_A4_rcmpneq(rs, rt)
}

/// `Rd32=!cmp.eq(Rs32,#s8)`
///
/// Instruction Type: ALU32_2op
/// Execution Slots: SLOT0123
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_not_cmp_eq_RI<const IS8: i32>(rs: i32) -> i32 {
    static_assert_simm_bits!(IS8, 8);
    hexagon_A4_rcmpneqi(rs, IS8)
}

/// `Rd32=round(Rs32,#u5)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(round, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_round_RI<const IU5: u32>(rs: i32) -> i32 {
    static_assert_uimm_bits!(IU5, 5);
    hexagon_A4_round_ri(rs, IU5 as i32)
}

/// `Rd32=round(Rs32,#u5):sat`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(round, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_round_RI_sat<const IU5: u32>(rs: i32) -> i32 {
    static_assert_uimm_bits!(IU5, 5);
    hexagon_A4_round_ri_sat(rs, IU5 as i32)
}

/// `Rd32=round(Rs32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(round))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_round_RR(rs: i32, rt: i32) -> i32 {
    hexagon_A4_round_rr(rs, rt)
}

/// `Rd32=round(Rs32,Rt32):sat`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(round))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_round_RR_sat(rs: i32, rt: i32) -> i32 {
    hexagon_A4_round_rr_sat(rs, rt)
}

/// `Pd4=tlbmatch(Rss32,Rt32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(tlbmatch))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_tlbmatch_PR(rss: i64, rt: i32) -> i32 {
    hexagon_A4_tlbmatch(rss, rt)
}

/// `Pd4=any8(vcmpb.eq(Rss32,Rtt32))`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(any8))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_any8_vcmpb_eq_PP(rss: i64, rtt: i64) -> i32 {
    hexagon_A4_vcmpbeq_any(rss, rtt)
}

/// `Pd4=vcmpb.eq(Rss32,#u8)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(vcmpb, IU8 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_vcmpb_eq_PI<const IU8: u32>(rss: i64) -> i32 {
    static_assert_uimm_bits!(IU8, 8);
    hexagon_A4_vcmpbeqi(rss, IU8 as i32)
}

/// `Pd4=vcmpb.gt(Rss32,Rtt32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vcmpb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_vcmpb_gt_PP(rss: i64, rtt: i64) -> i32 {
    hexagon_A4_vcmpbgt(rss, rtt)
}

/// `Pd4=vcmpb.gt(Rss32,#s8)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(vcmpb, IS8 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_vcmpb_gt_PI<const IS8: i32>(rss: i64) -> i32 {
    static_assert_simm_bits!(IS8, 8);
    hexagon_A4_vcmpbgti(rss, IS8)
}

/// `Pd4=vcmpb.gtu(Rss32,#u7)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(vcmpb, IU7 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_vcmpb_gtu_PI<const IU7: u32>(rss: i64) -> i32 {
    static_assert_uimm_bits!(IU7, 7);
    hexagon_A4_vcmpbgtui(rss, IU7 as i32)
}

/// `Pd4=vcmph.eq(Rss32,#s8)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(vcmph, IS8 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_vcmph_eq_PI<const IS8: i32>(rss: i64) -> i32 {
    static_assert_simm_bits!(IS8, 8);
    hexagon_A4_vcmpheqi(rss, IS8)
}

/// `Pd4=vcmph.gt(Rss32,#s8)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(vcmph, IS8 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_vcmph_gt_PI<const IS8: i32>(rss: i64) -> i32 {
    static_assert_simm_bits!(IS8, 8);
    hexagon_A4_vcmphgti(rss, IS8)
}

/// `Pd4=vcmph.gtu(Rss32,#u7)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(vcmph, IU7 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_vcmph_gtu_PI<const IU7: u32>(rss: i64) -> i32 {
    static_assert_uimm_bits!(IU7, 7);
    hexagon_A4_vcmphgtui(rss, IU7 as i32)
}

/// `Pd4=vcmpw.eq(Rss32,#s8)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(vcmpw, IS8 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_vcmpw_eq_PI<const IS8: i32>(rss: i64) -> i32 {
    static_assert_simm_bits!(IS8, 8);
    hexagon_A4_vcmpweqi(rss, IS8)
}

/// `Pd4=vcmpw.gt(Rss32,#s8)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(vcmpw, IS8 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_vcmpw_gt_PI<const IS8: i32>(rss: i64) -> i32 {
    static_assert_simm_bits!(IS8, 8);
    hexagon_A4_vcmpwgti(rss, IS8)
}

/// `Pd4=vcmpw.gtu(Rss32,#u7)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(vcmpw, IU7 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_vcmpw_gtu_PI<const IU7: u32>(rss: i64) -> i32 {
    static_assert_uimm_bits!(IU7, 7);
    hexagon_A4_vcmpwgtui(rss, IU7 as i32)
}

/// `Rxx32=vrmaxh(Rss32,Ru32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vrmaxh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vrmaxh_PR(rxx: i64, rss: i64, ru: i32) -> i64 {
    hexagon_A4_vrmaxh(rxx, rss, ru)
}

/// `Rxx32=vrmaxuh(Rss32,Ru32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vrmaxuh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vrmaxuh_PR(rxx: i64, rss: i64, ru: i32) -> i64 {
    hexagon_A4_vrmaxuh(rxx, rss, ru)
}

/// `Rxx32=vrmaxuw(Rss32,Ru32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vrmaxuw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vrmaxuw_PR(rxx: i64, rss: i64, ru: i32) -> i64 {
    hexagon_A4_vrmaxuw(rxx, rss, ru)
}

/// `Rxx32=vrmaxw(Rss32,Ru32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vrmaxw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vrmaxw_PR(rxx: i64, rss: i64, ru: i32) -> i64 {
    hexagon_A4_vrmaxw(rxx, rss, ru)
}

/// `Rxx32=vrminh(Rss32,Ru32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vrminh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vrminh_PR(rxx: i64, rss: i64, ru: i32) -> i64 {
    hexagon_A4_vrminh(rxx, rss, ru)
}

/// `Rxx32=vrminuh(Rss32,Ru32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vrminuh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vrminuh_PR(rxx: i64, rss: i64, ru: i32) -> i64 {
    hexagon_A4_vrminuh(rxx, rss, ru)
}

/// `Rxx32=vrminuw(Rss32,Ru32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vrminuw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vrminuw_PR(rxx: i64, rss: i64, ru: i32) -> i64 {
    hexagon_A4_vrminuw(rxx, rss, ru)
}

/// `Rxx32=vrminw(Rss32,Ru32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vrminw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vrminw_PR(rxx: i64, rss: i64, ru: i32) -> i64 {
    hexagon_A4_vrminw(rxx, rss, ru)
}

/// `Rd32=vaddhub(Rss32,Rtt32):sat`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vaddhub))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_vaddhub_PP_sat(rss: i64, rtt: i64) -> i32 {
    hexagon_A5_vaddhubs(rss, rtt)
}

/// `Pd4=all8(Ps4)`
///
/// Instruction Type: CR
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(all8))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_all8_p(ps: i32) -> i32 {
    hexagon_C2_all8(ps)
}

/// `Pd4=and(Pt4,Ps4)`
///
/// Instruction Type: CR
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(and))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_and_pp(pt: i32, ps: i32) -> i32 {
    hexagon_C2_and(pt, ps)
}

/// `Pd4=and(Pt4,!Ps4)`
///
/// Instruction Type: CR
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(and))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_and_pnp(pt: i32, ps: i32) -> i32 {
    hexagon_C2_andn(pt, ps)
}

/// `Pd4=any8(Ps4)`
///
/// Instruction Type: CR
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(any8))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_any8_p(ps: i32) -> i32 {
    hexagon_C2_any8(ps)
}

/// `Pd4=bitsclr(Rs32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(bitsclr))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_bitsclr_RR(rs: i32, rt: i32) -> i32 {
    hexagon_C2_bitsclr(rs, rt)
}

/// `Pd4=bitsclr(Rs32,#u6)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(bitsclr, IU6 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_bitsclr_RI<const IU6: u32>(rs: i32) -> i32 {
    static_assert_uimm_bits!(IU6, 6);
    hexagon_C2_bitsclri(rs, IU6 as i32)
}

/// `Pd4=bitsset(Rs32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(bitsset))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_bitsset_RR(rs: i32, rt: i32) -> i32 {
    hexagon_C2_bitsset(rs, rt)
}

/// `Pd4=cmp.eq(Rs32,Rt32)`
///
/// Instruction Type: ALU32_3op
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(test, assert_instr(cmp))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_cmp_eq_RR(rs: i32, rt: i32) -> i32 {
    hexagon_C2_cmpeq(rs, rt)
}

/// `Pd4=cmp.eq(Rs32,#s10)`
///
/// Instruction Type: ALU32_2op
/// Execution Slots: SLOT0123
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(cmp, IS10 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_cmp_eq_RI<const IS10: i32>(rs: i32) -> i32 {
    static_assert_simm_bits!(IS10, 10);
    hexagon_C2_cmpeqi(rs, IS10)
}

/// `Pd4=cmp.eq(Rss32,Rtt32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(cmp))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_cmp_eq_PP(rss: i64, rtt: i64) -> i32 {
    hexagon_C2_cmpeqp(rss, rtt)
}

/// `Pd4=cmp.ge(Rs32,#s8)`
///
/// Instruction Type: ALU32_2op
/// Execution Slots: SLOT0123
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(cmp, IS8 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_cmp_ge_RI<const IS8: i32>(rs: i32) -> i32 {
    static_assert_simm_bits!(IS8, 8);
    hexagon_C2_cmpgei(rs, IS8)
}

/// `Pd4=cmp.geu(Rs32,#u8)`
///
/// Instruction Type: ALU32_2op
/// Execution Slots: SLOT0123
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(cmp, IU8 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_cmp_geu_RI<const IU8: u32>(rs: i32) -> i32 {
    static_assert_uimm_bits!(IU8, 8);
    hexagon_C2_cmpgeui(rs, IU8 as i32)
}

/// `Pd4=cmp.gt(Rs32,Rt32)`
///
/// Instruction Type: ALU32_3op
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(test, assert_instr(cmp))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_cmp_gt_RR(rs: i32, rt: i32) -> i32 {
    hexagon_C2_cmpgt(rs, rt)
}

/// `Pd4=cmp.gt(Rs32,#s10)`
///
/// Instruction Type: ALU32_2op
/// Execution Slots: SLOT0123
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(cmp, IS10 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_cmp_gt_RI<const IS10: i32>(rs: i32) -> i32 {
    static_assert_simm_bits!(IS10, 10);
    hexagon_C2_cmpgti(rs, IS10)
}

/// `Pd4=cmp.gt(Rss32,Rtt32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(cmp))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_cmp_gt_PP(rss: i64, rtt: i64) -> i32 {
    hexagon_C2_cmpgtp(rss, rtt)
}

/// `Pd4=cmp.gtu(Rs32,Rt32)`
///
/// Instruction Type: ALU32_3op
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(test, assert_instr(cmp))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_cmp_gtu_RR(rs: i32, rt: i32) -> i32 {
    hexagon_C2_cmpgtu(rs, rt)
}

/// `Pd4=cmp.gtu(Rs32,#u9)`
///
/// Instruction Type: ALU32_2op
/// Execution Slots: SLOT0123
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(cmp, IU9 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_cmp_gtu_RI<const IU9: u32>(rs: i32) -> i32 {
    static_assert_uimm_bits!(IU9, 9);
    hexagon_C2_cmpgtui(rs, IU9 as i32)
}

/// `Pd4=cmp.gtu(Rss32,Rtt32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(cmp))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_cmp_gtu_PP(rss: i64, rtt: i64) -> i32 {
    hexagon_C2_cmpgtup(rss, rtt)
}

/// `Pd4=cmp.lt(Rs32,Rt32)`
///
/// Instruction Type: ALU32_3op
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(test, assert_instr(cmp))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_cmp_lt_RR(rs: i32, rt: i32) -> i32 {
    hexagon_C2_cmplt(rs, rt)
}

/// `Pd4=cmp.ltu(Rs32,Rt32)`
///
/// Instruction Type: ALU32_3op
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(test, assert_instr(cmp))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_cmp_ltu_RR(rs: i32, rt: i32) -> i32 {
    hexagon_C2_cmpltu(rs, rt)
}

/// `Rdd32=mask(Pt4)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mask))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mask_p(pt: i32) -> i64 {
    hexagon_C2_mask(pt)
}

/// `Rd32=mux(Pu4,Rs32,Rt32)`
///
/// Instruction Type: ALU32_3op
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(test, assert_instr(mux))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mux_pRR(pu: i32, rs: i32, rt: i32) -> i32 {
    hexagon_C2_mux(pu, rs, rt)
}

/// `Rd32=mux(Pu4,#s8,#S8)`
///
/// Instruction Type: ALU32_2op
/// Execution Slots: SLOT0123
#[inline(always)]
#[rustc_legacy_const_generics(1, 2)]
#[cfg_attr(test, assert_instr(mux, IS8 = 0, IS8_2 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mux_pII<const IS8: i32, const IS8_2: i32>(pu: i32) -> i32 {
    static_assert_simm_bits!(IS8, 8);
    static_assert_simm_bits!(IS8_2, 8);
    hexagon_C2_muxii(pu, IS8, IS8_2)
}

/// `Rd32=mux(Pu4,Rs32,#s8)`
///
/// Instruction Type: ALU32_2op
/// Execution Slots: SLOT0123
#[inline(always)]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(mux, IS8 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mux_pRI<const IS8: i32>(pu: i32, rs: i32) -> i32 {
    static_assert_simm_bits!(IS8, 8);
    hexagon_C2_muxir(pu, rs, IS8)
}

/// `Rd32=mux(Pu4,#s8,Rs32)`
///
/// Instruction Type: ALU32_2op
/// Execution Slots: SLOT0123
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(mux, IS8 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mux_pIR<const IS8: i32>(pu: i32, rs: i32) -> i32 {
    static_assert_simm_bits!(IS8, 8);
    hexagon_C2_muxri(pu, IS8, rs)
}

/// `Pd4=not(Ps4)`
///
/// Instruction Type: CR
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(not))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_not_p(ps: i32) -> i32 {
    hexagon_C2_not(ps)
}

/// `Pd4=or(Pt4,Ps4)`
///
/// Instruction Type: CR
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(or))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_or_pp(pt: i32, ps: i32) -> i32 {
    hexagon_C2_or(pt, ps)
}

/// `Pd4=or(Pt4,!Ps4)`
///
/// Instruction Type: CR
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(or))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_or_pnp(pt: i32, ps: i32) -> i32 {
    hexagon_C2_orn(pt, ps)
}

/// `Pd4=Ps4`
///
/// Instruction Type: MAPPING
/// Execution Slots: SLOT0123
#[inline(always)]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_equals_p(ps: i32) -> i32 {
    hexagon_C2_pxfer_map(ps)
}

/// `Rd32=Ps4`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_equals_p(ps: i32) -> i32 {
    hexagon_C2_tfrpr(ps)
}

/// `Pd4=Rs32`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_equals_R(rs: i32) -> i32 {
    hexagon_C2_tfrrp(rs)
}

/// `Rd32=vitpack(Ps4,Pt4)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vitpack))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_vitpack_pp(ps: i32, pt: i32) -> i32 {
    hexagon_C2_vitpack(ps, pt)
}

/// `Rdd32=vmux(Pu4,Rss32,Rtt32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmux))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmux_pPP(pu: i32, rss: i64, rtt: i64) -> i64 {
    hexagon_C2_vmux(pu, rss, rtt)
}

/// `Pd4=xor(Ps4,Pt4)`
///
/// Instruction Type: CR
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(xor))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_xor_pp(ps: i32, pt: i32) -> i32 {
    hexagon_C2_xor(ps, pt)
}

/// `Pd4=and(Ps4,and(Pt4,Pu4))`
///
/// Instruction Type: CR
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(and))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_and_and_ppp(ps: i32, pt: i32, pu: i32) -> i32 {
    hexagon_C4_and_and(ps, pt, pu)
}

/// `Pd4=and(Ps4,and(Pt4,!Pu4))`
///
/// Instruction Type: CR
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(and))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_and_and_ppnp(ps: i32, pt: i32, pu: i32) -> i32 {
    hexagon_C4_and_andn(ps, pt, pu)
}

/// `Pd4=and(Ps4,or(Pt4,Pu4))`
///
/// Instruction Type: CR
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(and))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_and_or_ppp(ps: i32, pt: i32, pu: i32) -> i32 {
    hexagon_C4_and_or(ps, pt, pu)
}

/// `Pd4=and(Ps4,or(Pt4,!Pu4))`
///
/// Instruction Type: CR
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(and))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_and_or_ppnp(ps: i32, pt: i32, pu: i32) -> i32 {
    hexagon_C4_and_orn(ps, pt, pu)
}

/// `Pd4=!cmp.gt(Rs32,Rt32)`
///
/// Instruction Type: ALU32_3op
/// Execution Slots: SLOT0123
#[inline(always)]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_not_cmp_gt_RR(rs: i32, rt: i32) -> i32 {
    hexagon_C4_cmplte(rs, rt)
}

/// `Pd4=!cmp.gt(Rs32,#s10)`
///
/// Instruction Type: ALU32_2op
/// Execution Slots: SLOT0123
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_not_cmp_gt_RI<const IS10: i32>(rs: i32) -> i32 {
    static_assert_simm_bits!(IS10, 10);
    hexagon_C4_cmpltei(rs, IS10)
}

/// `Pd4=!cmp.gtu(Rs32,Rt32)`
///
/// Instruction Type: ALU32_3op
/// Execution Slots: SLOT0123
#[inline(always)]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_not_cmp_gtu_RR(rs: i32, rt: i32) -> i32 {
    hexagon_C4_cmplteu(rs, rt)
}

/// `Pd4=!cmp.gtu(Rs32,#u9)`
///
/// Instruction Type: ALU32_2op
/// Execution Slots: SLOT0123
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_not_cmp_gtu_RI<const IU9: u32>(rs: i32) -> i32 {
    static_assert_uimm_bits!(IU9, 9);
    hexagon_C4_cmplteui(rs, IU9 as i32)
}

/// `Pd4=!cmp.eq(Rs32,Rt32)`
///
/// Instruction Type: ALU32_3op
/// Execution Slots: SLOT0123
#[inline(always)]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_not_cmp_eq_RR(rs: i32, rt: i32) -> i32 {
    hexagon_C4_cmpneq(rs, rt)
}

/// `Pd4=!cmp.eq(Rs32,#s10)`
///
/// Instruction Type: ALU32_2op
/// Execution Slots: SLOT0123
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_not_cmp_eq_RI<const IS10: i32>(rs: i32) -> i32 {
    static_assert_simm_bits!(IS10, 10);
    hexagon_C4_cmpneqi(rs, IS10)
}

/// `Pd4=fastcorner9(Ps4,Pt4)`
///
/// Instruction Type: CR
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(fastcorner9))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_fastcorner9_pp(ps: i32, pt: i32) -> i32 {
    hexagon_C4_fastcorner9(ps, pt)
}

/// `Pd4=!fastcorner9(Ps4,Pt4)`
///
/// Instruction Type: CR
/// Execution Slots: SLOT23
#[inline(always)]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_not_fastcorner9_pp(ps: i32, pt: i32) -> i32 {
    hexagon_C4_fastcorner9_not(ps, pt)
}

/// `Pd4=!bitsclr(Rs32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_not_bitsclr_RR(rs: i32, rt: i32) -> i32 {
    hexagon_C4_nbitsclr(rs, rt)
}

/// `Pd4=!bitsclr(Rs32,#u6)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_not_bitsclr_RI<const IU6: u32>(rs: i32) -> i32 {
    static_assert_uimm_bits!(IU6, 6);
    hexagon_C4_nbitsclri(rs, IU6 as i32)
}

/// `Pd4=!bitsset(Rs32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_not_bitsset_RR(rs: i32, rt: i32) -> i32 {
    hexagon_C4_nbitsset(rs, rt)
}

/// `Pd4=or(Ps4,and(Pt4,Pu4))`
///
/// Instruction Type: CR
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(or))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_or_and_ppp(ps: i32, pt: i32, pu: i32) -> i32 {
    hexagon_C4_or_and(ps, pt, pu)
}

/// `Pd4=or(Ps4,and(Pt4,!Pu4))`
///
/// Instruction Type: CR
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(or))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_or_and_ppnp(ps: i32, pt: i32, pu: i32) -> i32 {
    hexagon_C4_or_andn(ps, pt, pu)
}

/// `Pd4=or(Ps4,or(Pt4,Pu4))`
///
/// Instruction Type: CR
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(or))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_or_or_ppp(ps: i32, pt: i32, pu: i32) -> i32 {
    hexagon_C4_or_or(ps, pt, pu)
}

/// `Pd4=or(Ps4,or(Pt4,!Pu4))`
///
/// Instruction Type: CR
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(or))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_or_or_ppnp(ps: i32, pt: i32, pu: i32) -> i32 {
    hexagon_C4_or_orn(ps, pt, pu)
}

/// `Rdd32=convert_d2df(Rss32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(convert_d2df))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_convert_d2df_P(rss: i64) -> f64 {
    hexagon_F2_conv_d2df(rss)
}

/// `Rd32=convert_d2sf(Rss32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(convert_d2sf))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_convert_d2sf_P(rss: i64) -> f32 {
    hexagon_F2_conv_d2sf(rss)
}

/// `Rdd32=convert_df2d(Rss32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(convert_df2d))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_convert_df2d_P(rss: f64) -> i64 {
    hexagon_F2_conv_df2d(rss)
}

/// `Rdd32=convert_df2d(Rss32):chop`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(convert_df2d))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_convert_df2d_P_chop(rss: f64) -> i64 {
    hexagon_F2_conv_df2d_chop(rss)
}

/// `Rd32=convert_df2sf(Rss32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(convert_df2sf))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_convert_df2sf_P(rss: f64) -> f32 {
    hexagon_F2_conv_df2sf(rss)
}

/// `Rdd32=convert_df2ud(Rss32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(convert_df2ud))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_convert_df2ud_P(rss: f64) -> i64 {
    hexagon_F2_conv_df2ud(rss)
}

/// `Rdd32=convert_df2ud(Rss32):chop`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(convert_df2ud))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_convert_df2ud_P_chop(rss: f64) -> i64 {
    hexagon_F2_conv_df2ud_chop(rss)
}

/// `Rd32=convert_df2uw(Rss32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(convert_df2uw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_convert_df2uw_P(rss: f64) -> i32 {
    hexagon_F2_conv_df2uw(rss)
}

/// `Rd32=convert_df2uw(Rss32):chop`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(convert_df2uw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_convert_df2uw_P_chop(rss: f64) -> i32 {
    hexagon_F2_conv_df2uw_chop(rss)
}

/// `Rd32=convert_df2w(Rss32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(convert_df2w))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_convert_df2w_P(rss: f64) -> i32 {
    hexagon_F2_conv_df2w(rss)
}

/// `Rd32=convert_df2w(Rss32):chop`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(convert_df2w))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_convert_df2w_P_chop(rss: f64) -> i32 {
    hexagon_F2_conv_df2w_chop(rss)
}

/// `Rdd32=convert_sf2d(Rs32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(convert_sf2d))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_convert_sf2d_R(rs: f32) -> i64 {
    hexagon_F2_conv_sf2d(rs)
}

/// `Rdd32=convert_sf2d(Rs32):chop`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(convert_sf2d))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_convert_sf2d_R_chop(rs: f32) -> i64 {
    hexagon_F2_conv_sf2d_chop(rs)
}

/// `Rdd32=convert_sf2df(Rs32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(convert_sf2df))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_convert_sf2df_R(rs: f32) -> f64 {
    hexagon_F2_conv_sf2df(rs)
}

/// `Rdd32=convert_sf2ud(Rs32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(convert_sf2ud))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_convert_sf2ud_R(rs: f32) -> i64 {
    hexagon_F2_conv_sf2ud(rs)
}

/// `Rdd32=convert_sf2ud(Rs32):chop`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(convert_sf2ud))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_convert_sf2ud_R_chop(rs: f32) -> i64 {
    hexagon_F2_conv_sf2ud_chop(rs)
}

/// `Rd32=convert_sf2uw(Rs32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(convert_sf2uw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_convert_sf2uw_R(rs: f32) -> i32 {
    hexagon_F2_conv_sf2uw(rs)
}

/// `Rd32=convert_sf2uw(Rs32):chop`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(convert_sf2uw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_convert_sf2uw_R_chop(rs: f32) -> i32 {
    hexagon_F2_conv_sf2uw_chop(rs)
}

/// `Rd32=convert_sf2w(Rs32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(convert_sf2w))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_convert_sf2w_R(rs: f32) -> i32 {
    hexagon_F2_conv_sf2w(rs)
}

/// `Rd32=convert_sf2w(Rs32):chop`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(convert_sf2w))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_convert_sf2w_R_chop(rs: f32) -> i32 {
    hexagon_F2_conv_sf2w_chop(rs)
}

/// `Rdd32=convert_ud2df(Rss32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(convert_ud2df))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_convert_ud2df_P(rss: i64) -> f64 {
    hexagon_F2_conv_ud2df(rss)
}

/// `Rd32=convert_ud2sf(Rss32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(convert_ud2sf))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_convert_ud2sf_P(rss: i64) -> f32 {
    hexagon_F2_conv_ud2sf(rss)
}

/// `Rdd32=convert_uw2df(Rs32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(convert_uw2df))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_convert_uw2df_R(rs: i32) -> f64 {
    hexagon_F2_conv_uw2df(rs)
}

/// `Rd32=convert_uw2sf(Rs32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(convert_uw2sf))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_convert_uw2sf_R(rs: i32) -> f32 {
    hexagon_F2_conv_uw2sf(rs)
}

/// `Rdd32=convert_w2df(Rs32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(convert_w2df))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_convert_w2df_R(rs: i32) -> f64 {
    hexagon_F2_conv_w2df(rs)
}

/// `Rd32=convert_w2sf(Rs32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(convert_w2sf))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_convert_w2sf_R(rs: i32) -> f32 {
    hexagon_F2_conv_w2sf(rs)
}

/// `Pd4=dfclass(Rss32,#u5)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(dfclass, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_dfclass_PI<const IU5: u32>(rss: f64) -> i32 {
    static_assert_uimm_bits!(IU5, 5);
    hexagon_F2_dfclass(rss, IU5 as i32)
}

/// `Pd4=dfcmp.eq(Rss32,Rtt32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(dfcmp))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_dfcmp_eq_PP(rss: f64, rtt: f64) -> i32 {
    hexagon_F2_dfcmpeq(rss, rtt)
}

/// `Pd4=dfcmp.ge(Rss32,Rtt32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(dfcmp))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_dfcmp_ge_PP(rss: f64, rtt: f64) -> i32 {
    hexagon_F2_dfcmpge(rss, rtt)
}

/// `Pd4=dfcmp.gt(Rss32,Rtt32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(dfcmp))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_dfcmp_gt_PP(rss: f64, rtt: f64) -> i32 {
    hexagon_F2_dfcmpgt(rss, rtt)
}

/// `Pd4=dfcmp.uo(Rss32,Rtt32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(dfcmp))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_dfcmp_uo_PP(rss: f64, rtt: f64) -> i32 {
    hexagon_F2_dfcmpuo(rss, rtt)
}

/// `Rdd32=dfmake(#u10):neg`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(0)]
#[cfg_attr(test, assert_instr(dfmake, IU10 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_dfmake_I_neg<const IU10: u32>() -> f64 {
    static_assert_uimm_bits!(IU10, 10);
    hexagon_F2_dfimm_n(IU10 as i32)
}

/// `Rdd32=dfmake(#u10):pos`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(0)]
#[cfg_attr(test, assert_instr(dfmake, IU10 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_dfmake_I_pos<const IU10: u32>() -> f64 {
    static_assert_uimm_bits!(IU10, 10);
    hexagon_F2_dfimm_p(IU10 as i32)
}

/// `Rd32=sfadd(Rs32,Rt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(sfadd))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_sfadd_RR(rs: f32, rt: f32) -> f32 {
    hexagon_F2_sfadd(rs, rt)
}

/// `Pd4=sfclass(Rs32,#u5)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(sfclass, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_sfclass_RI<const IU5: u32>(rs: f32) -> i32 {
    static_assert_uimm_bits!(IU5, 5);
    hexagon_F2_sfclass(rs, IU5 as i32)
}

/// `Pd4=sfcmp.eq(Rs32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(sfcmp))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_sfcmp_eq_RR(rs: f32, rt: f32) -> i32 {
    hexagon_F2_sfcmpeq(rs, rt)
}

/// `Pd4=sfcmp.ge(Rs32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(sfcmp))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_sfcmp_ge_RR(rs: f32, rt: f32) -> i32 {
    hexagon_F2_sfcmpge(rs, rt)
}

/// `Pd4=sfcmp.gt(Rs32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(sfcmp))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_sfcmp_gt_RR(rs: f32, rt: f32) -> i32 {
    hexagon_F2_sfcmpgt(rs, rt)
}

/// `Pd4=sfcmp.uo(Rs32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(sfcmp))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_sfcmp_uo_RR(rs: f32, rt: f32) -> i32 {
    hexagon_F2_sfcmpuo(rs, rt)
}

/// `Rd32=sffixupd(Rs32,Rt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(sffixupd))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_sffixupd_RR(rs: f32, rt: f32) -> f32 {
    hexagon_F2_sffixupd(rs, rt)
}

/// `Rd32=sffixupn(Rs32,Rt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(sffixupn))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_sffixupn_RR(rs: f32, rt: f32) -> f32 {
    hexagon_F2_sffixupn(rs, rt)
}

/// `Rd32=sffixupr(Rs32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(sffixupr))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_sffixupr_R(rs: f32) -> f32 {
    hexagon_F2_sffixupr(rs)
}

/// `Rx32+=sfmpy(Rs32,Rt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(sfmpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_sfmpyacc_RR(rx: f32, rs: f32, rt: f32) -> f32 {
    hexagon_F2_sffma(rx, rs, rt)
}

/// `Rx32+=sfmpy(Rs32,Rt32):lib`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(sfmpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_sfmpyacc_RR_lib(rx: f32, rs: f32, rt: f32) -> f32 {
    hexagon_F2_sffma_lib(rx, rs, rt)
}

/// `Rx32+=sfmpy(Rs32,Rt32,Pu4):scale`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(sfmpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_sfmpyacc_RRp_scale(rx: f32, rs: f32, rt: f32, pu: i32) -> f32 {
    hexagon_F2_sffma_sc(rx, rs, rt, pu)
}

/// `Rx32-=sfmpy(Rs32,Rt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(sfmpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_sfmpynac_RR(rx: f32, rs: f32, rt: f32) -> f32 {
    hexagon_F2_sffms(rx, rs, rt)
}

/// `Rx32-=sfmpy(Rs32,Rt32):lib`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(sfmpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_sfmpynac_RR_lib(rx: f32, rs: f32, rt: f32) -> f32 {
    hexagon_F2_sffms_lib(rx, rs, rt)
}

/// `Rd32=sfmake(#u10):neg`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(0)]
#[cfg_attr(test, assert_instr(sfmake, IU10 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_sfmake_I_neg<const IU10: u32>() -> f32 {
    static_assert_uimm_bits!(IU10, 10);
    hexagon_F2_sfimm_n(IU10 as i32)
}

/// `Rd32=sfmake(#u10):pos`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(0)]
#[cfg_attr(test, assert_instr(sfmake, IU10 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_sfmake_I_pos<const IU10: u32>() -> f32 {
    static_assert_uimm_bits!(IU10, 10);
    hexagon_F2_sfimm_p(IU10 as i32)
}

/// `Rd32=sfmax(Rs32,Rt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(sfmax))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_sfmax_RR(rs: f32, rt: f32) -> f32 {
    hexagon_F2_sfmax(rs, rt)
}

/// `Rd32=sfmin(Rs32,Rt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(sfmin))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_sfmin_RR(rs: f32, rt: f32) -> f32 {
    hexagon_F2_sfmin(rs, rt)
}

/// `Rd32=sfmpy(Rs32,Rt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(sfmpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_sfmpy_RR(rs: f32, rt: f32) -> f32 {
    hexagon_F2_sfmpy(rs, rt)
}

/// `Rd32=sfsub(Rs32,Rt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(sfsub))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_sfsub_RR(rs: f32, rt: f32) -> f32 {
    hexagon_F2_sfsub(rs, rt)
}

/// `Rx32+=add(Rs32,Rt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(add))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_addacc_RR(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M2_acci(rx, rs, rt)
}

/// `Rx32+=add(Rs32,#s8)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(add, IS8 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_addacc_RI<const IS8: i32>(rx: i32, rs: i32) -> i32 {
    static_assert_simm_bits!(IS8, 8);
    hexagon_M2_accii(rx, rs, IS8)
}

/// `Rxx32+=cmpyi(Rs32,Rt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(cmpyi))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_cmpyiacc_RR(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M2_cmaci_s0(rxx, rs, rt)
}

/// `Rxx32+=cmpyr(Rs32,Rt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(cmpyr))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_cmpyracc_RR(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M2_cmacr_s0(rxx, rs, rt)
}

/// `Rxx32+=cmpy(Rs32,Rt32):sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(cmpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_cmpyacc_RR_sat(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M2_cmacs_s0(rxx, rs, rt)
}

/// `Rxx32+=cmpy(Rs32,Rt32):<<1:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(cmpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_cmpyacc_RR_s1_sat(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M2_cmacs_s1(rxx, rs, rt)
}

/// `Rxx32+=cmpy(Rs32,Rt32*):sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(cmpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_cmpyacc_RR_conj_sat(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M2_cmacsc_s0(rxx, rs, rt)
}

/// `Rxx32+=cmpy(Rs32,Rt32*):<<1:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(cmpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_cmpyacc_RR_conj_s1_sat(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M2_cmacsc_s1(rxx, rs, rt)
}

/// `Rdd32=cmpyi(Rs32,Rt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(cmpyi))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_cmpyi_RR(rs: i32, rt: i32) -> i64 {
    hexagon_M2_cmpyi_s0(rs, rt)
}

/// `Rdd32=cmpyr(Rs32,Rt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(cmpyr))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_cmpyr_RR(rs: i32, rt: i32) -> i64 {
    hexagon_M2_cmpyr_s0(rs, rt)
}

/// `Rd32=cmpy(Rs32,Rt32):rnd:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(cmpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_cmpy_RR_rnd_sat(rs: i32, rt: i32) -> i32 {
    hexagon_M2_cmpyrs_s0(rs, rt)
}

/// `Rd32=cmpy(Rs32,Rt32):<<1:rnd:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(cmpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_cmpy_RR_s1_rnd_sat(rs: i32, rt: i32) -> i32 {
    hexagon_M2_cmpyrs_s1(rs, rt)
}

/// `Rd32=cmpy(Rs32,Rt32*):rnd:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(cmpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_cmpy_RR_conj_rnd_sat(rs: i32, rt: i32) -> i32 {
    hexagon_M2_cmpyrsc_s0(rs, rt)
}

/// `Rd32=cmpy(Rs32,Rt32*):<<1:rnd:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(cmpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_cmpy_RR_conj_s1_rnd_sat(rs: i32, rt: i32) -> i32 {
    hexagon_M2_cmpyrsc_s1(rs, rt)
}

/// `Rdd32=cmpy(Rs32,Rt32):sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(cmpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_cmpy_RR_sat(rs: i32, rt: i32) -> i64 {
    hexagon_M2_cmpys_s0(rs, rt)
}

/// `Rdd32=cmpy(Rs32,Rt32):<<1:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(cmpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_cmpy_RR_s1_sat(rs: i32, rt: i32) -> i64 {
    hexagon_M2_cmpys_s1(rs, rt)
}

/// `Rdd32=cmpy(Rs32,Rt32*):sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(cmpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_cmpy_RR_conj_sat(rs: i32, rt: i32) -> i64 {
    hexagon_M2_cmpysc_s0(rs, rt)
}

/// `Rdd32=cmpy(Rs32,Rt32*):<<1:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(cmpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_cmpy_RR_conj_s1_sat(rs: i32, rt: i32) -> i64 {
    hexagon_M2_cmpysc_s1(rs, rt)
}

/// `Rxx32-=cmpy(Rs32,Rt32):sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(cmpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_cmpynac_RR_sat(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M2_cnacs_s0(rxx, rs, rt)
}

/// `Rxx32-=cmpy(Rs32,Rt32):<<1:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(cmpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_cmpynac_RR_s1_sat(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M2_cnacs_s1(rxx, rs, rt)
}

/// `Rxx32-=cmpy(Rs32,Rt32*):sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(cmpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_cmpynac_RR_conj_sat(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M2_cnacsc_s0(rxx, rs, rt)
}

/// `Rxx32-=cmpy(Rs32,Rt32*):<<1:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(cmpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_cmpynac_RR_conj_s1_sat(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M2_cnacsc_s1(rxx, rs, rt)
}

/// `Rxx32+=mpy(Rs32,Rt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpyacc_RR(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M2_dpmpyss_acc_s0(rxx, rs, rt)
}

/// `Rxx32-=mpy(Rs32,Rt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpynac_RR(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M2_dpmpyss_nac_s0(rxx, rs, rt)
}

/// `Rd32=mpy(Rs32,Rt32):rnd`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpy_RR_rnd(rs: i32, rt: i32) -> i32 {
    hexagon_M2_dpmpyss_rnd_s0(rs, rt)
}

/// `Rdd32=mpy(Rs32,Rt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpy_RR(rs: i32, rt: i32) -> i64 {
    hexagon_M2_dpmpyss_s0(rs, rt)
}

/// `Rxx32+=mpyu(Rs32,Rt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpyuacc_RR(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M2_dpmpyuu_acc_s0(rxx, rs, rt)
}

/// `Rxx32-=mpyu(Rs32,Rt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpyunac_RR(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M2_dpmpyuu_nac_s0(rxx, rs, rt)
}

/// `Rdd32=mpyu(Rs32,Rt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpyu_RR(rs: i32, rt: i32) -> i64 {
    hexagon_M2_dpmpyuu_s0(rs, rt)
}

/// `Rd32=mpy(Rs32,Rt32.h):<<1:rnd:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpy_RRh_s1_rnd_sat(rs: i32, rt: i32) -> i32 {
    hexagon_M2_hmmpyh_rs1(rs, rt)
}

/// `Rd32=mpy(Rs32,Rt32.h):<<1:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpy_RRh_s1_sat(rs: i32, rt: i32) -> i32 {
    hexagon_M2_hmmpyh_s1(rs, rt)
}

/// `Rd32=mpy(Rs32,Rt32.l):<<1:rnd:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpy_RRl_s1_rnd_sat(rs: i32, rt: i32) -> i32 {
    hexagon_M2_hmmpyl_rs1(rs, rt)
}

/// `Rd32=mpy(Rs32,Rt32.l):<<1:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpy_RRl_s1_sat(rs: i32, rt: i32) -> i32 {
    hexagon_M2_hmmpyl_s1(rs, rt)
}

/// `Rx32+=mpyi(Rs32,Rt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyi))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpyiacc_RR(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M2_maci(rx, rs, rt)
}

/// `Rx32-=mpyi(Rs32,#u8)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(mpyi, IU8 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpyinac_RI<const IU8: u32>(rx: i32, rs: i32) -> i32 {
    static_assert_uimm_bits!(IU8, 8);
    hexagon_M2_macsin(rx, rs, IU8 as i32)
}

/// `Rx32+=mpyi(Rs32,#u8)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(mpyi, IU8 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpyiacc_RI<const IU8: u32>(rx: i32, rs: i32) -> i32 {
    static_assert_uimm_bits!(IU8, 8);
    hexagon_M2_macsip(rx, rs, IU8 as i32)
}

/// `Rxx32+=vmpywoh(Rss32,Rtt32):rnd:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmpywoh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmpywohacc_PP_rnd_sat(rxx: i64, rss: i64, rtt: i64) -> i64 {
    hexagon_M2_mmachs_rs0(rxx, rss, rtt)
}

/// `Rxx32+=vmpywoh(Rss32,Rtt32):<<1:rnd:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmpywoh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmpywohacc_PP_s1_rnd_sat(rxx: i64, rss: i64, rtt: i64) -> i64 {
    hexagon_M2_mmachs_rs1(rxx, rss, rtt)
}

/// `Rxx32+=vmpywoh(Rss32,Rtt32):sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmpywoh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmpywohacc_PP_sat(rxx: i64, rss: i64, rtt: i64) -> i64 {
    hexagon_M2_mmachs_s0(rxx, rss, rtt)
}

/// `Rxx32+=vmpywoh(Rss32,Rtt32):<<1:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmpywoh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmpywohacc_PP_s1_sat(rxx: i64, rss: i64, rtt: i64) -> i64 {
    hexagon_M2_mmachs_s1(rxx, rss, rtt)
}

/// `Rxx32+=vmpyweh(Rss32,Rtt32):rnd:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmpyweh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmpywehacc_PP_rnd_sat(rxx: i64, rss: i64, rtt: i64) -> i64 {
    hexagon_M2_mmacls_rs0(rxx, rss, rtt)
}

/// `Rxx32+=vmpyweh(Rss32,Rtt32):<<1:rnd:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmpyweh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmpywehacc_PP_s1_rnd_sat(rxx: i64, rss: i64, rtt: i64) -> i64 {
    hexagon_M2_mmacls_rs1(rxx, rss, rtt)
}

/// `Rxx32+=vmpyweh(Rss32,Rtt32):sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmpyweh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmpywehacc_PP_sat(rxx: i64, rss: i64, rtt: i64) -> i64 {
    hexagon_M2_mmacls_s0(rxx, rss, rtt)
}

/// `Rxx32+=vmpyweh(Rss32,Rtt32):<<1:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmpyweh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmpywehacc_PP_s1_sat(rxx: i64, rss: i64, rtt: i64) -> i64 {
    hexagon_M2_mmacls_s1(rxx, rss, rtt)
}

/// `Rxx32+=vmpywouh(Rss32,Rtt32):rnd:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmpywouh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmpywouhacc_PP_rnd_sat(rxx: i64, rss: i64, rtt: i64) -> i64 {
    hexagon_M2_mmacuhs_rs0(rxx, rss, rtt)
}

/// `Rxx32+=vmpywouh(Rss32,Rtt32):<<1:rnd:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmpywouh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmpywouhacc_PP_s1_rnd_sat(rxx: i64, rss: i64, rtt: i64) -> i64 {
    hexagon_M2_mmacuhs_rs1(rxx, rss, rtt)
}

/// `Rxx32+=vmpywouh(Rss32,Rtt32):sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmpywouh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmpywouhacc_PP_sat(rxx: i64, rss: i64, rtt: i64) -> i64 {
    hexagon_M2_mmacuhs_s0(rxx, rss, rtt)
}

/// `Rxx32+=vmpywouh(Rss32,Rtt32):<<1:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmpywouh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmpywouhacc_PP_s1_sat(rxx: i64, rss: i64, rtt: i64) -> i64 {
    hexagon_M2_mmacuhs_s1(rxx, rss, rtt)
}

/// `Rxx32+=vmpyweuh(Rss32,Rtt32):rnd:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmpyweuh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmpyweuhacc_PP_rnd_sat(rxx: i64, rss: i64, rtt: i64) -> i64 {
    hexagon_M2_mmaculs_rs0(rxx, rss, rtt)
}

/// `Rxx32+=vmpyweuh(Rss32,Rtt32):<<1:rnd:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmpyweuh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmpyweuhacc_PP_s1_rnd_sat(rxx: i64, rss: i64, rtt: i64) -> i64 {
    hexagon_M2_mmaculs_rs1(rxx, rss, rtt)
}

/// `Rxx32+=vmpyweuh(Rss32,Rtt32):sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmpyweuh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmpyweuhacc_PP_sat(rxx: i64, rss: i64, rtt: i64) -> i64 {
    hexagon_M2_mmaculs_s0(rxx, rss, rtt)
}

/// `Rxx32+=vmpyweuh(Rss32,Rtt32):<<1:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmpyweuh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmpyweuhacc_PP_s1_sat(rxx: i64, rss: i64, rtt: i64) -> i64 {
    hexagon_M2_mmaculs_s1(rxx, rss, rtt)
}

/// `Rdd32=vmpywoh(Rss32,Rtt32):rnd:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmpywoh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmpywoh_PP_rnd_sat(rss: i64, rtt: i64) -> i64 {
    hexagon_M2_mmpyh_rs0(rss, rtt)
}

/// `Rdd32=vmpywoh(Rss32,Rtt32):<<1:rnd:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmpywoh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmpywoh_PP_s1_rnd_sat(rss: i64, rtt: i64) -> i64 {
    hexagon_M2_mmpyh_rs1(rss, rtt)
}

/// `Rdd32=vmpywoh(Rss32,Rtt32):sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmpywoh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmpywoh_PP_sat(rss: i64, rtt: i64) -> i64 {
    hexagon_M2_mmpyh_s0(rss, rtt)
}

/// `Rdd32=vmpywoh(Rss32,Rtt32):<<1:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmpywoh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmpywoh_PP_s1_sat(rss: i64, rtt: i64) -> i64 {
    hexagon_M2_mmpyh_s1(rss, rtt)
}

/// `Rdd32=vmpyweh(Rss32,Rtt32):rnd:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmpyweh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmpyweh_PP_rnd_sat(rss: i64, rtt: i64) -> i64 {
    hexagon_M2_mmpyl_rs0(rss, rtt)
}

/// `Rdd32=vmpyweh(Rss32,Rtt32):<<1:rnd:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmpyweh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmpyweh_PP_s1_rnd_sat(rss: i64, rtt: i64) -> i64 {
    hexagon_M2_mmpyl_rs1(rss, rtt)
}

/// `Rdd32=vmpyweh(Rss32,Rtt32):sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmpyweh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmpyweh_PP_sat(rss: i64, rtt: i64) -> i64 {
    hexagon_M2_mmpyl_s0(rss, rtt)
}

/// `Rdd32=vmpyweh(Rss32,Rtt32):<<1:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmpyweh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmpyweh_PP_s1_sat(rss: i64, rtt: i64) -> i64 {
    hexagon_M2_mmpyl_s1(rss, rtt)
}

/// `Rdd32=vmpywouh(Rss32,Rtt32):rnd:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmpywouh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmpywouh_PP_rnd_sat(rss: i64, rtt: i64) -> i64 {
    hexagon_M2_mmpyuh_rs0(rss, rtt)
}

/// `Rdd32=vmpywouh(Rss32,Rtt32):<<1:rnd:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmpywouh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmpywouh_PP_s1_rnd_sat(rss: i64, rtt: i64) -> i64 {
    hexagon_M2_mmpyuh_rs1(rss, rtt)
}

/// `Rdd32=vmpywouh(Rss32,Rtt32):sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmpywouh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmpywouh_PP_sat(rss: i64, rtt: i64) -> i64 {
    hexagon_M2_mmpyuh_s0(rss, rtt)
}

/// `Rdd32=vmpywouh(Rss32,Rtt32):<<1:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmpywouh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmpywouh_PP_s1_sat(rss: i64, rtt: i64) -> i64 {
    hexagon_M2_mmpyuh_s1(rss, rtt)
}

/// `Rdd32=vmpyweuh(Rss32,Rtt32):rnd:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmpyweuh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmpyweuh_PP_rnd_sat(rss: i64, rtt: i64) -> i64 {
    hexagon_M2_mmpyul_rs0(rss, rtt)
}

/// `Rdd32=vmpyweuh(Rss32,Rtt32):<<1:rnd:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmpyweuh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmpyweuh_PP_s1_rnd_sat(rss: i64, rtt: i64) -> i64 {
    hexagon_M2_mmpyul_rs1(rss, rtt)
}

/// `Rdd32=vmpyweuh(Rss32,Rtt32):sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmpyweuh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmpyweuh_PP_sat(rss: i64, rtt: i64) -> i64 {
    hexagon_M2_mmpyul_s0(rss, rtt)
}

/// `Rdd32=vmpyweuh(Rss32,Rtt32):<<1:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmpyweuh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmpyweuh_PP_s1_sat(rss: i64, rtt: i64) -> i64 {
    hexagon_M2_mmpyul_s1(rss, rtt)
}

/// `Rx32+=mpy(Rs32.h,Rt32.h)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpyacc_RhRh(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_acc_hh_s0(rx, rs, rt)
}

/// `Rx32+=mpy(Rs32.h,Rt32.h):<<1`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpyacc_RhRh_s1(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_acc_hh_s1(rx, rs, rt)
}

/// `Rx32+=mpy(Rs32.h,Rt32.l)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpyacc_RhRl(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_acc_hl_s0(rx, rs, rt)
}

/// `Rx32+=mpy(Rs32.h,Rt32.l):<<1`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpyacc_RhRl_s1(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_acc_hl_s1(rx, rs, rt)
}

/// `Rx32+=mpy(Rs32.l,Rt32.h)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpyacc_RlRh(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_acc_lh_s0(rx, rs, rt)
}

/// `Rx32+=mpy(Rs32.l,Rt32.h):<<1`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpyacc_RlRh_s1(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_acc_lh_s1(rx, rs, rt)
}

/// `Rx32+=mpy(Rs32.l,Rt32.l)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpyacc_RlRl(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_acc_ll_s0(rx, rs, rt)
}

/// `Rx32+=mpy(Rs32.l,Rt32.l):<<1`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpyacc_RlRl_s1(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_acc_ll_s1(rx, rs, rt)
}

/// `Rx32+=mpy(Rs32.h,Rt32.h):sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpyacc_RhRh_sat(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_acc_sat_hh_s0(rx, rs, rt)
}

/// `Rx32+=mpy(Rs32.h,Rt32.h):<<1:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpyacc_RhRh_s1_sat(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_acc_sat_hh_s1(rx, rs, rt)
}

/// `Rx32+=mpy(Rs32.h,Rt32.l):sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpyacc_RhRl_sat(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_acc_sat_hl_s0(rx, rs, rt)
}

/// `Rx32+=mpy(Rs32.h,Rt32.l):<<1:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpyacc_RhRl_s1_sat(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_acc_sat_hl_s1(rx, rs, rt)
}

/// `Rx32+=mpy(Rs32.l,Rt32.h):sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpyacc_RlRh_sat(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_acc_sat_lh_s0(rx, rs, rt)
}

/// `Rx32+=mpy(Rs32.l,Rt32.h):<<1:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpyacc_RlRh_s1_sat(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_acc_sat_lh_s1(rx, rs, rt)
}

/// `Rx32+=mpy(Rs32.l,Rt32.l):sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpyacc_RlRl_sat(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_acc_sat_ll_s0(rx, rs, rt)
}

/// `Rx32+=mpy(Rs32.l,Rt32.l):<<1:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpyacc_RlRl_s1_sat(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_acc_sat_ll_s1(rx, rs, rt)
}

/// `Rd32=mpy(Rs32.h,Rt32.h)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpy_RhRh(rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_hh_s0(rs, rt)
}

/// `Rd32=mpy(Rs32.h,Rt32.h):<<1`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpy_RhRh_s1(rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_hh_s1(rs, rt)
}

/// `Rd32=mpy(Rs32.h,Rt32.l)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpy_RhRl(rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_hl_s0(rs, rt)
}

/// `Rd32=mpy(Rs32.h,Rt32.l):<<1`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpy_RhRl_s1(rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_hl_s1(rs, rt)
}

/// `Rd32=mpy(Rs32.l,Rt32.h)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpy_RlRh(rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_lh_s0(rs, rt)
}

/// `Rd32=mpy(Rs32.l,Rt32.h):<<1`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpy_RlRh_s1(rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_lh_s1(rs, rt)
}

/// `Rd32=mpy(Rs32.l,Rt32.l)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpy_RlRl(rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_ll_s0(rs, rt)
}

/// `Rd32=mpy(Rs32.l,Rt32.l):<<1`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpy_RlRl_s1(rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_ll_s1(rs, rt)
}

/// `Rx32-=mpy(Rs32.h,Rt32.h)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpynac_RhRh(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_nac_hh_s0(rx, rs, rt)
}

/// `Rx32-=mpy(Rs32.h,Rt32.h):<<1`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpynac_RhRh_s1(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_nac_hh_s1(rx, rs, rt)
}

/// `Rx32-=mpy(Rs32.h,Rt32.l)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpynac_RhRl(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_nac_hl_s0(rx, rs, rt)
}

/// `Rx32-=mpy(Rs32.h,Rt32.l):<<1`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpynac_RhRl_s1(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_nac_hl_s1(rx, rs, rt)
}

/// `Rx32-=mpy(Rs32.l,Rt32.h)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpynac_RlRh(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_nac_lh_s0(rx, rs, rt)
}

/// `Rx32-=mpy(Rs32.l,Rt32.h):<<1`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpynac_RlRh_s1(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_nac_lh_s1(rx, rs, rt)
}

/// `Rx32-=mpy(Rs32.l,Rt32.l)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpynac_RlRl(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_nac_ll_s0(rx, rs, rt)
}

/// `Rx32-=mpy(Rs32.l,Rt32.l):<<1`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpynac_RlRl_s1(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_nac_ll_s1(rx, rs, rt)
}

/// `Rx32-=mpy(Rs32.h,Rt32.h):sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpynac_RhRh_sat(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_nac_sat_hh_s0(rx, rs, rt)
}

/// `Rx32-=mpy(Rs32.h,Rt32.h):<<1:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpynac_RhRh_s1_sat(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_nac_sat_hh_s1(rx, rs, rt)
}

/// `Rx32-=mpy(Rs32.h,Rt32.l):sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpynac_RhRl_sat(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_nac_sat_hl_s0(rx, rs, rt)
}

/// `Rx32-=mpy(Rs32.h,Rt32.l):<<1:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpynac_RhRl_s1_sat(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_nac_sat_hl_s1(rx, rs, rt)
}

/// `Rx32-=mpy(Rs32.l,Rt32.h):sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpynac_RlRh_sat(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_nac_sat_lh_s0(rx, rs, rt)
}

/// `Rx32-=mpy(Rs32.l,Rt32.h):<<1:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpynac_RlRh_s1_sat(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_nac_sat_lh_s1(rx, rs, rt)
}

/// `Rx32-=mpy(Rs32.l,Rt32.l):sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpynac_RlRl_sat(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_nac_sat_ll_s0(rx, rs, rt)
}

/// `Rx32-=mpy(Rs32.l,Rt32.l):<<1:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpynac_RlRl_s1_sat(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_nac_sat_ll_s1(rx, rs, rt)
}

/// `Rd32=mpy(Rs32.h,Rt32.h):rnd`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpy_RhRh_rnd(rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_rnd_hh_s0(rs, rt)
}

/// `Rd32=mpy(Rs32.h,Rt32.h):<<1:rnd`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpy_RhRh_s1_rnd(rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_rnd_hh_s1(rs, rt)
}

/// `Rd32=mpy(Rs32.h,Rt32.l):rnd`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpy_RhRl_rnd(rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_rnd_hl_s0(rs, rt)
}

/// `Rd32=mpy(Rs32.h,Rt32.l):<<1:rnd`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpy_RhRl_s1_rnd(rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_rnd_hl_s1(rs, rt)
}

/// `Rd32=mpy(Rs32.l,Rt32.h):rnd`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpy_RlRh_rnd(rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_rnd_lh_s0(rs, rt)
}

/// `Rd32=mpy(Rs32.l,Rt32.h):<<1:rnd`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpy_RlRh_s1_rnd(rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_rnd_lh_s1(rs, rt)
}

/// `Rd32=mpy(Rs32.l,Rt32.l):rnd`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpy_RlRl_rnd(rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_rnd_ll_s0(rs, rt)
}

/// `Rd32=mpy(Rs32.l,Rt32.l):<<1:rnd`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpy_RlRl_s1_rnd(rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_rnd_ll_s1(rs, rt)
}

/// `Rd32=mpy(Rs32.h,Rt32.h):sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpy_RhRh_sat(rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_sat_hh_s0(rs, rt)
}

/// `Rd32=mpy(Rs32.h,Rt32.h):<<1:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpy_RhRh_s1_sat(rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_sat_hh_s1(rs, rt)
}

/// `Rd32=mpy(Rs32.h,Rt32.l):sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpy_RhRl_sat(rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_sat_hl_s0(rs, rt)
}

/// `Rd32=mpy(Rs32.h,Rt32.l):<<1:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpy_RhRl_s1_sat(rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_sat_hl_s1(rs, rt)
}

/// `Rd32=mpy(Rs32.l,Rt32.h):sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpy_RlRh_sat(rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_sat_lh_s0(rs, rt)
}

/// `Rd32=mpy(Rs32.l,Rt32.h):<<1:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpy_RlRh_s1_sat(rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_sat_lh_s1(rs, rt)
}

/// `Rd32=mpy(Rs32.l,Rt32.l):sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpy_RlRl_sat(rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_sat_ll_s0(rs, rt)
}

/// `Rd32=mpy(Rs32.l,Rt32.l):<<1:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpy_RlRl_s1_sat(rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_sat_ll_s1(rs, rt)
}

/// `Rd32=mpy(Rs32.h,Rt32.h):rnd:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpy_RhRh_rnd_sat(rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_sat_rnd_hh_s0(rs, rt)
}

/// `Rd32=mpy(Rs32.h,Rt32.h):<<1:rnd:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpy_RhRh_s1_rnd_sat(rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_sat_rnd_hh_s1(rs, rt)
}

/// `Rd32=mpy(Rs32.h,Rt32.l):rnd:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpy_RhRl_rnd_sat(rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_sat_rnd_hl_s0(rs, rt)
}

/// `Rd32=mpy(Rs32.h,Rt32.l):<<1:rnd:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpy_RhRl_s1_rnd_sat(rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_sat_rnd_hl_s1(rs, rt)
}

/// `Rd32=mpy(Rs32.l,Rt32.h):rnd:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpy_RlRh_rnd_sat(rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_sat_rnd_lh_s0(rs, rt)
}

/// `Rd32=mpy(Rs32.l,Rt32.h):<<1:rnd:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpy_RlRh_s1_rnd_sat(rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_sat_rnd_lh_s1(rs, rt)
}

/// `Rd32=mpy(Rs32.l,Rt32.l):rnd:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpy_RlRl_rnd_sat(rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_sat_rnd_ll_s0(rs, rt)
}

/// `Rd32=mpy(Rs32.l,Rt32.l):<<1:rnd:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpy_RlRl_s1_rnd_sat(rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_sat_rnd_ll_s1(rs, rt)
}

/// `Rd32=mpy(Rs32,Rt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpy_RR(rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_up(rs, rt)
}

/// `Rd32=mpy(Rs32,Rt32):<<1`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpy_RR_s1(rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_up_s1(rs, rt)
}

/// `Rd32=mpy(Rs32,Rt32):<<1:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpy_RR_s1_sat(rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpy_up_s1_sat(rs, rt)
}

/// `Rxx32+=mpy(Rs32.h,Rt32.h)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpyacc_RhRh(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyd_acc_hh_s0(rxx, rs, rt)
}

/// `Rxx32+=mpy(Rs32.h,Rt32.h):<<1`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpyacc_RhRh_s1(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyd_acc_hh_s1(rxx, rs, rt)
}

/// `Rxx32+=mpy(Rs32.h,Rt32.l)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpyacc_RhRl(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyd_acc_hl_s0(rxx, rs, rt)
}

/// `Rxx32+=mpy(Rs32.h,Rt32.l):<<1`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpyacc_RhRl_s1(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyd_acc_hl_s1(rxx, rs, rt)
}

/// `Rxx32+=mpy(Rs32.l,Rt32.h)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpyacc_RlRh(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyd_acc_lh_s0(rxx, rs, rt)
}

/// `Rxx32+=mpy(Rs32.l,Rt32.h):<<1`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpyacc_RlRh_s1(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyd_acc_lh_s1(rxx, rs, rt)
}

/// `Rxx32+=mpy(Rs32.l,Rt32.l)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpyacc_RlRl(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyd_acc_ll_s0(rxx, rs, rt)
}

/// `Rxx32+=mpy(Rs32.l,Rt32.l):<<1`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpyacc_RlRl_s1(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyd_acc_ll_s1(rxx, rs, rt)
}

/// `Rdd32=mpy(Rs32.h,Rt32.h)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpy_RhRh(rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyd_hh_s0(rs, rt)
}

/// `Rdd32=mpy(Rs32.h,Rt32.h):<<1`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpy_RhRh_s1(rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyd_hh_s1(rs, rt)
}

/// `Rdd32=mpy(Rs32.h,Rt32.l)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpy_RhRl(rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyd_hl_s0(rs, rt)
}

/// `Rdd32=mpy(Rs32.h,Rt32.l):<<1`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpy_RhRl_s1(rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyd_hl_s1(rs, rt)
}

/// `Rdd32=mpy(Rs32.l,Rt32.h)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpy_RlRh(rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyd_lh_s0(rs, rt)
}

/// `Rdd32=mpy(Rs32.l,Rt32.h):<<1`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpy_RlRh_s1(rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyd_lh_s1(rs, rt)
}

/// `Rdd32=mpy(Rs32.l,Rt32.l)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpy_RlRl(rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyd_ll_s0(rs, rt)
}

/// `Rdd32=mpy(Rs32.l,Rt32.l):<<1`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpy_RlRl_s1(rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyd_ll_s1(rs, rt)
}

/// `Rxx32-=mpy(Rs32.h,Rt32.h)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpynac_RhRh(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyd_nac_hh_s0(rxx, rs, rt)
}

/// `Rxx32-=mpy(Rs32.h,Rt32.h):<<1`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpynac_RhRh_s1(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyd_nac_hh_s1(rxx, rs, rt)
}

/// `Rxx32-=mpy(Rs32.h,Rt32.l)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpynac_RhRl(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyd_nac_hl_s0(rxx, rs, rt)
}

/// `Rxx32-=mpy(Rs32.h,Rt32.l):<<1`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpynac_RhRl_s1(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyd_nac_hl_s1(rxx, rs, rt)
}

/// `Rxx32-=mpy(Rs32.l,Rt32.h)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpynac_RlRh(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyd_nac_lh_s0(rxx, rs, rt)
}

/// `Rxx32-=mpy(Rs32.l,Rt32.h):<<1`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpynac_RlRh_s1(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyd_nac_lh_s1(rxx, rs, rt)
}

/// `Rxx32-=mpy(Rs32.l,Rt32.l)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpynac_RlRl(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyd_nac_ll_s0(rxx, rs, rt)
}

/// `Rxx32-=mpy(Rs32.l,Rt32.l):<<1`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpynac_RlRl_s1(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyd_nac_ll_s1(rxx, rs, rt)
}

/// `Rdd32=mpy(Rs32.h,Rt32.h):rnd`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpy_RhRh_rnd(rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyd_rnd_hh_s0(rs, rt)
}

/// `Rdd32=mpy(Rs32.h,Rt32.h):<<1:rnd`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpy_RhRh_s1_rnd(rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyd_rnd_hh_s1(rs, rt)
}

/// `Rdd32=mpy(Rs32.h,Rt32.l):rnd`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpy_RhRl_rnd(rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyd_rnd_hl_s0(rs, rt)
}

/// `Rdd32=mpy(Rs32.h,Rt32.l):<<1:rnd`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpy_RhRl_s1_rnd(rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyd_rnd_hl_s1(rs, rt)
}

/// `Rdd32=mpy(Rs32.l,Rt32.h):rnd`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpy_RlRh_rnd(rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyd_rnd_lh_s0(rs, rt)
}

/// `Rdd32=mpy(Rs32.l,Rt32.h):<<1:rnd`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpy_RlRh_s1_rnd(rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyd_rnd_lh_s1(rs, rt)
}

/// `Rdd32=mpy(Rs32.l,Rt32.l):rnd`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpy_RlRl_rnd(rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyd_rnd_ll_s0(rs, rt)
}

/// `Rdd32=mpy(Rs32.l,Rt32.l):<<1:rnd`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpy_RlRl_s1_rnd(rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyd_rnd_ll_s1(rs, rt)
}

/// `Rd32=mpyi(Rs32,Rt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyi))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpyi_RR(rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpyi(rs, rt)
}

/// `Rd32=mpyi(Rs32,#m9)`
///
/// Instruction Type: M
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyi))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpyi_RI(rs: i32, im9: i32) -> i32 {
    hexagon_M2_mpysmi(rs, im9)
}

/// `Rd32=mpysu(Rs32,Rt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpysu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpysu_RR(rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpysu_up(rs, rt)
}

/// `Rx32+=mpyu(Rs32.h,Rt32.h)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpyuacc_RhRh(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpyu_acc_hh_s0(rx, rs, rt)
}

/// `Rx32+=mpyu(Rs32.h,Rt32.h):<<1`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpyuacc_RhRh_s1(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpyu_acc_hh_s1(rx, rs, rt)
}

/// `Rx32+=mpyu(Rs32.h,Rt32.l)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpyuacc_RhRl(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpyu_acc_hl_s0(rx, rs, rt)
}

/// `Rx32+=mpyu(Rs32.h,Rt32.l):<<1`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpyuacc_RhRl_s1(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpyu_acc_hl_s1(rx, rs, rt)
}

/// `Rx32+=mpyu(Rs32.l,Rt32.h)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpyuacc_RlRh(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpyu_acc_lh_s0(rx, rs, rt)
}

/// `Rx32+=mpyu(Rs32.l,Rt32.h):<<1`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpyuacc_RlRh_s1(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpyu_acc_lh_s1(rx, rs, rt)
}

/// `Rx32+=mpyu(Rs32.l,Rt32.l)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpyuacc_RlRl(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpyu_acc_ll_s0(rx, rs, rt)
}

/// `Rx32+=mpyu(Rs32.l,Rt32.l):<<1`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpyuacc_RlRl_s1(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpyu_acc_ll_s1(rx, rs, rt)
}

/// `Rd32=mpyu(Rs32.h,Rt32.h)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpyu_RhRh(rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpyu_hh_s0(rs, rt)
}

/// `Rd32=mpyu(Rs32.h,Rt32.h):<<1`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpyu_RhRh_s1(rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpyu_hh_s1(rs, rt)
}

/// `Rd32=mpyu(Rs32.h,Rt32.l)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpyu_RhRl(rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpyu_hl_s0(rs, rt)
}

/// `Rd32=mpyu(Rs32.h,Rt32.l):<<1`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpyu_RhRl_s1(rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpyu_hl_s1(rs, rt)
}

/// `Rd32=mpyu(Rs32.l,Rt32.h)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpyu_RlRh(rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpyu_lh_s0(rs, rt)
}

/// `Rd32=mpyu(Rs32.l,Rt32.h):<<1`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpyu_RlRh_s1(rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpyu_lh_s1(rs, rt)
}

/// `Rd32=mpyu(Rs32.l,Rt32.l)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpyu_RlRl(rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpyu_ll_s0(rs, rt)
}

/// `Rd32=mpyu(Rs32.l,Rt32.l):<<1`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpyu_RlRl_s1(rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpyu_ll_s1(rs, rt)
}

/// `Rx32-=mpyu(Rs32.h,Rt32.h)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpyunac_RhRh(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpyu_nac_hh_s0(rx, rs, rt)
}

/// `Rx32-=mpyu(Rs32.h,Rt32.h):<<1`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpyunac_RhRh_s1(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpyu_nac_hh_s1(rx, rs, rt)
}

/// `Rx32-=mpyu(Rs32.h,Rt32.l)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpyunac_RhRl(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpyu_nac_hl_s0(rx, rs, rt)
}

/// `Rx32-=mpyu(Rs32.h,Rt32.l):<<1`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpyunac_RhRl_s1(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpyu_nac_hl_s1(rx, rs, rt)
}

/// `Rx32-=mpyu(Rs32.l,Rt32.h)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpyunac_RlRh(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpyu_nac_lh_s0(rx, rs, rt)
}

/// `Rx32-=mpyu(Rs32.l,Rt32.h):<<1`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpyunac_RlRh_s1(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpyu_nac_lh_s1(rx, rs, rt)
}

/// `Rx32-=mpyu(Rs32.l,Rt32.l)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpyunac_RlRl(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpyu_nac_ll_s0(rx, rs, rt)
}

/// `Rx32-=mpyu(Rs32.l,Rt32.l):<<1`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpyunac_RlRl_s1(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpyu_nac_ll_s1(rx, rs, rt)
}

/// `Rd32=mpyu(Rs32,Rt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpyu_RR(rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpyu_up(rs, rt)
}

/// `Rxx32+=mpyu(Rs32.h,Rt32.h)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpyuacc_RhRh(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyud_acc_hh_s0(rxx, rs, rt)
}

/// `Rxx32+=mpyu(Rs32.h,Rt32.h):<<1`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpyuacc_RhRh_s1(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyud_acc_hh_s1(rxx, rs, rt)
}

/// `Rxx32+=mpyu(Rs32.h,Rt32.l)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpyuacc_RhRl(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyud_acc_hl_s0(rxx, rs, rt)
}

/// `Rxx32+=mpyu(Rs32.h,Rt32.l):<<1`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpyuacc_RhRl_s1(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyud_acc_hl_s1(rxx, rs, rt)
}

/// `Rxx32+=mpyu(Rs32.l,Rt32.h)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpyuacc_RlRh(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyud_acc_lh_s0(rxx, rs, rt)
}

/// `Rxx32+=mpyu(Rs32.l,Rt32.h):<<1`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpyuacc_RlRh_s1(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyud_acc_lh_s1(rxx, rs, rt)
}

/// `Rxx32+=mpyu(Rs32.l,Rt32.l)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpyuacc_RlRl(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyud_acc_ll_s0(rxx, rs, rt)
}

/// `Rxx32+=mpyu(Rs32.l,Rt32.l):<<1`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpyuacc_RlRl_s1(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyud_acc_ll_s1(rxx, rs, rt)
}

/// `Rdd32=mpyu(Rs32.h,Rt32.h)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpyu_RhRh(rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyud_hh_s0(rs, rt)
}

/// `Rdd32=mpyu(Rs32.h,Rt32.h):<<1`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpyu_RhRh_s1(rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyud_hh_s1(rs, rt)
}

/// `Rdd32=mpyu(Rs32.h,Rt32.l)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpyu_RhRl(rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyud_hl_s0(rs, rt)
}

/// `Rdd32=mpyu(Rs32.h,Rt32.l):<<1`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpyu_RhRl_s1(rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyud_hl_s1(rs, rt)
}

/// `Rdd32=mpyu(Rs32.l,Rt32.h)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpyu_RlRh(rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyud_lh_s0(rs, rt)
}

/// `Rdd32=mpyu(Rs32.l,Rt32.h):<<1`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpyu_RlRh_s1(rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyud_lh_s1(rs, rt)
}

/// `Rdd32=mpyu(Rs32.l,Rt32.l)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpyu_RlRl(rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyud_ll_s0(rs, rt)
}

/// `Rdd32=mpyu(Rs32.l,Rt32.l):<<1`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpyu_RlRl_s1(rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyud_ll_s1(rs, rt)
}

/// `Rxx32-=mpyu(Rs32.h,Rt32.h)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpyunac_RhRh(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyud_nac_hh_s0(rxx, rs, rt)
}

/// `Rxx32-=mpyu(Rs32.h,Rt32.h):<<1`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpyunac_RhRh_s1(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyud_nac_hh_s1(rxx, rs, rt)
}

/// `Rxx32-=mpyu(Rs32.h,Rt32.l)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpyunac_RhRl(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyud_nac_hl_s0(rxx, rs, rt)
}

/// `Rxx32-=mpyu(Rs32.h,Rt32.l):<<1`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpyunac_RhRl_s1(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyud_nac_hl_s1(rxx, rs, rt)
}

/// `Rxx32-=mpyu(Rs32.l,Rt32.h)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpyunac_RlRh(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyud_nac_lh_s0(rxx, rs, rt)
}

/// `Rxx32-=mpyu(Rs32.l,Rt32.h):<<1`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpyunac_RlRh_s1(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyud_nac_lh_s1(rxx, rs, rt)
}

/// `Rxx32-=mpyu(Rs32.l,Rt32.l)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpyunac_RlRl(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyud_nac_ll_s0(rxx, rs, rt)
}

/// `Rxx32-=mpyu(Rs32.l,Rt32.l):<<1`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_mpyunac_RlRl_s1(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M2_mpyud_nac_ll_s1(rxx, rs, rt)
}

/// `Rd32=mpyui(Rs32,Rt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(test, assert_instr(mpyui))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpyui_RR(rs: i32, rt: i32) -> i32 {
    hexagon_M2_mpyui(rs, rt)
}

/// `Rx32-=add(Rs32,Rt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(add))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_addnac_RR(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M2_nacci(rx, rs, rt)
}

/// `Rx32-=add(Rs32,#s8)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(add, IS8 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_addnac_RI<const IS8: i32>(rx: i32, rs: i32) -> i32 {
    static_assert_simm_bits!(IS8, 8);
    hexagon_M2_naccii(rx, rs, IS8)
}

/// `Rx32+=sub(Rt32,Rs32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(sub))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_subacc_RR(rx: i32, rt: i32, rs: i32) -> i32 {
    hexagon_M2_subacc(rx, rt, rs)
}

/// `Rdd32=vabsdiffh(Rtt32,Rss32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vabsdiffh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vabsdiffh_PP(rtt: i64, rss: i64) -> i64 {
    hexagon_M2_vabsdiffh(rtt, rss)
}

/// `Rdd32=vabsdiffw(Rtt32,Rss32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vabsdiffw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vabsdiffw_PP(rtt: i64, rss: i64) -> i64 {
    hexagon_M2_vabsdiffw(rtt, rss)
}

/// `Rxx32+=vcmpyi(Rss32,Rtt32):sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vcmpyi))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vcmpyiacc_PP_sat(rxx: i64, rss: i64, rtt: i64) -> i64 {
    hexagon_M2_vcmac_s0_sat_i(rxx, rss, rtt)
}

/// `Rxx32+=vcmpyr(Rss32,Rtt32):sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vcmpyr))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vcmpyracc_PP_sat(rxx: i64, rss: i64, rtt: i64) -> i64 {
    hexagon_M2_vcmac_s0_sat_r(rxx, rss, rtt)
}

/// `Rdd32=vcmpyi(Rss32,Rtt32):sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vcmpyi))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vcmpyi_PP_sat(rss: i64, rtt: i64) -> i64 {
    hexagon_M2_vcmpy_s0_sat_i(rss, rtt)
}

/// `Rdd32=vcmpyr(Rss32,Rtt32):sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vcmpyr))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vcmpyr_PP_sat(rss: i64, rtt: i64) -> i64 {
    hexagon_M2_vcmpy_s0_sat_r(rss, rtt)
}

/// `Rdd32=vcmpyi(Rss32,Rtt32):<<1:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vcmpyi))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vcmpyi_PP_s1_sat(rss: i64, rtt: i64) -> i64 {
    hexagon_M2_vcmpy_s1_sat_i(rss, rtt)
}

/// `Rdd32=vcmpyr(Rss32,Rtt32):<<1:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vcmpyr))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vcmpyr_PP_s1_sat(rss: i64, rtt: i64) -> i64 {
    hexagon_M2_vcmpy_s1_sat_r(rss, rtt)
}

/// `Rxx32+=vdmpy(Rss32,Rtt32):sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vdmpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vdmpyacc_PP_sat(rxx: i64, rss: i64, rtt: i64) -> i64 {
    hexagon_M2_vdmacs_s0(rxx, rss, rtt)
}

/// `Rxx32+=vdmpy(Rss32,Rtt32):<<1:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vdmpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vdmpyacc_PP_s1_sat(rxx: i64, rss: i64, rtt: i64) -> i64 {
    hexagon_M2_vdmacs_s1(rxx, rss, rtt)
}

/// `Rd32=vdmpy(Rss32,Rtt32):rnd:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vdmpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_vdmpy_PP_rnd_sat(rss: i64, rtt: i64) -> i32 {
    hexagon_M2_vdmpyrs_s0(rss, rtt)
}

/// `Rd32=vdmpy(Rss32,Rtt32):<<1:rnd:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vdmpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_vdmpy_PP_s1_rnd_sat(rss: i64, rtt: i64) -> i32 {
    hexagon_M2_vdmpyrs_s1(rss, rtt)
}

/// `Rdd32=vdmpy(Rss32,Rtt32):sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vdmpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vdmpy_PP_sat(rss: i64, rtt: i64) -> i64 {
    hexagon_M2_vdmpys_s0(rss, rtt)
}

/// `Rdd32=vdmpy(Rss32,Rtt32):<<1:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vdmpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vdmpy_PP_s1_sat(rss: i64, rtt: i64) -> i64 {
    hexagon_M2_vdmpys_s1(rss, rtt)
}

/// `Rxx32+=vmpyh(Rs32,Rt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmpyh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmpyhacc_RR(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M2_vmac2(rxx, rs, rt)
}

/// `Rxx32+=vmpyeh(Rss32,Rtt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmpyeh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmpyehacc_PP(rxx: i64, rss: i64, rtt: i64) -> i64 {
    hexagon_M2_vmac2es(rxx, rss, rtt)
}

/// `Rxx32+=vmpyeh(Rss32,Rtt32):sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmpyeh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmpyehacc_PP_sat(rxx: i64, rss: i64, rtt: i64) -> i64 {
    hexagon_M2_vmac2es_s0(rxx, rss, rtt)
}

/// `Rxx32+=vmpyeh(Rss32,Rtt32):<<1:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmpyeh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmpyehacc_PP_s1_sat(rxx: i64, rss: i64, rtt: i64) -> i64 {
    hexagon_M2_vmac2es_s1(rxx, rss, rtt)
}

/// `Rxx32+=vmpyh(Rs32,Rt32):sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmpyh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmpyhacc_RR_sat(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M2_vmac2s_s0(rxx, rs, rt)
}

/// `Rxx32+=vmpyh(Rs32,Rt32):<<1:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmpyh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmpyhacc_RR_s1_sat(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M2_vmac2s_s1(rxx, rs, rt)
}

/// `Rxx32+=vmpyhsu(Rs32,Rt32):sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmpyhsu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmpyhsuacc_RR_sat(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M2_vmac2su_s0(rxx, rs, rt)
}

/// `Rxx32+=vmpyhsu(Rs32,Rt32):<<1:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmpyhsu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmpyhsuacc_RR_s1_sat(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M2_vmac2su_s1(rxx, rs, rt)
}

/// `Rdd32=vmpyeh(Rss32,Rtt32):sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmpyeh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmpyeh_PP_sat(rss: i64, rtt: i64) -> i64 {
    hexagon_M2_vmpy2es_s0(rss, rtt)
}

/// `Rdd32=vmpyeh(Rss32,Rtt32):<<1:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmpyeh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmpyeh_PP_s1_sat(rss: i64, rtt: i64) -> i64 {
    hexagon_M2_vmpy2es_s1(rss, rtt)
}

/// `Rdd32=vmpyh(Rs32,Rt32):sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmpyh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmpyh_RR_sat(rs: i32, rt: i32) -> i64 {
    hexagon_M2_vmpy2s_s0(rs, rt)
}

/// `Rd32=vmpyh(Rs32,Rt32):rnd:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmpyh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_vmpyh_RR_rnd_sat(rs: i32, rt: i32) -> i32 {
    hexagon_M2_vmpy2s_s0pack(rs, rt)
}

/// `Rdd32=vmpyh(Rs32,Rt32):<<1:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmpyh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmpyh_RR_s1_sat(rs: i32, rt: i32) -> i64 {
    hexagon_M2_vmpy2s_s1(rs, rt)
}

/// `Rd32=vmpyh(Rs32,Rt32):<<1:rnd:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmpyh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_vmpyh_RR_s1_rnd_sat(rs: i32, rt: i32) -> i32 {
    hexagon_M2_vmpy2s_s1pack(rs, rt)
}

/// `Rdd32=vmpyhsu(Rs32,Rt32):sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmpyhsu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmpyhsu_RR_sat(rs: i32, rt: i32) -> i64 {
    hexagon_M2_vmpy2su_s0(rs, rt)
}

/// `Rdd32=vmpyhsu(Rs32,Rt32):<<1:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmpyhsu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmpyhsu_RR_s1_sat(rs: i32, rt: i32) -> i64 {
    hexagon_M2_vmpy2su_s1(rs, rt)
}

/// `Rd32=vraddh(Rss32,Rtt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vraddh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_vraddh_PP(rss: i64, rtt: i64) -> i32 {
    hexagon_M2_vraddh(rss, rtt)
}

/// `Rd32=vradduh(Rss32,Rtt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vradduh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_vradduh_PP(rss: i64, rtt: i64) -> i32 {
    hexagon_M2_vradduh(rss, rtt)
}

/// `Rxx32+=vrcmpyi(Rss32,Rtt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vrcmpyi))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vrcmpyiacc_PP(rxx: i64, rss: i64, rtt: i64) -> i64 {
    hexagon_M2_vrcmaci_s0(rxx, rss, rtt)
}

/// `Rxx32+=vrcmpyi(Rss32,Rtt32*)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vrcmpyi))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vrcmpyiacc_PP_conj(rxx: i64, rss: i64, rtt: i64) -> i64 {
    hexagon_M2_vrcmaci_s0c(rxx, rss, rtt)
}

/// `Rxx32+=vrcmpyr(Rss32,Rtt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vrcmpyr))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vrcmpyracc_PP(rxx: i64, rss: i64, rtt: i64) -> i64 {
    hexagon_M2_vrcmacr_s0(rxx, rss, rtt)
}

/// `Rxx32+=vrcmpyr(Rss32,Rtt32*)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vrcmpyr))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vrcmpyracc_PP_conj(rxx: i64, rss: i64, rtt: i64) -> i64 {
    hexagon_M2_vrcmacr_s0c(rxx, rss, rtt)
}

/// `Rdd32=vrcmpyi(Rss32,Rtt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vrcmpyi))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vrcmpyi_PP(rss: i64, rtt: i64) -> i64 {
    hexagon_M2_vrcmpyi_s0(rss, rtt)
}

/// `Rdd32=vrcmpyi(Rss32,Rtt32*)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vrcmpyi))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vrcmpyi_PP_conj(rss: i64, rtt: i64) -> i64 {
    hexagon_M2_vrcmpyi_s0c(rss, rtt)
}

/// `Rdd32=vrcmpyr(Rss32,Rtt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vrcmpyr))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vrcmpyr_PP(rss: i64, rtt: i64) -> i64 {
    hexagon_M2_vrcmpyr_s0(rss, rtt)
}

/// `Rdd32=vrcmpyr(Rss32,Rtt32*)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vrcmpyr))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vrcmpyr_PP_conj(rss: i64, rtt: i64) -> i64 {
    hexagon_M2_vrcmpyr_s0c(rss, rtt)
}

/// `Rxx32+=vrcmpys(Rss32,Rt32):<<1:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(test, assert_instr(vrcmpys))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vrcmpysacc_PR_s1_sat(rxx: i64, rss: i64, rt: i32) -> i64 {
    hexagon_M2_vrcmpys_acc_s1(rxx, rss, rt)
}

/// `Rdd32=vrcmpys(Rss32,Rt32):<<1:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(test, assert_instr(vrcmpys))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vrcmpys_PR_s1_sat(rss: i64, rt: i32) -> i64 {
    hexagon_M2_vrcmpys_s1(rss, rt)
}

/// `Rd32=vrcmpys(Rss32,Rt32):<<1:rnd:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(test, assert_instr(vrcmpys))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_vrcmpys_PR_s1_rnd_sat(rss: i64, rt: i32) -> i32 {
    hexagon_M2_vrcmpys_s1rp(rss, rt)
}

/// `Rxx32+=vrmpyh(Rss32,Rtt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vrmpyh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vrmpyhacc_PP(rxx: i64, rss: i64, rtt: i64) -> i64 {
    hexagon_M2_vrmac_s0(rxx, rss, rtt)
}

/// `Rdd32=vrmpyh(Rss32,Rtt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vrmpyh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vrmpyh_PP(rss: i64, rtt: i64) -> i64 {
    hexagon_M2_vrmpy_s0(rss, rtt)
}

/// `Rx32^=xor(Rs32,Rt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(xor))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_xorxacc_RR(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M2_xor_xacc(rx, rs, rt)
}

/// `Rx32&=and(Rs32,Rt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(and))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_andand_RR(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M4_and_and(rx, rs, rt)
}

/// `Rx32&=and(Rs32,~Rt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(and))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_andand_RnR(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M4_and_andn(rx, rs, rt)
}

/// `Rx32&=or(Rs32,Rt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(or))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_orand_RR(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M4_and_or(rx, rs, rt)
}

/// `Rx32&=xor(Rs32,Rt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(xor))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_xorand_RR(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M4_and_xor(rx, rs, rt)
}

/// `Rd32=cmpyiwh(Rss32,Rt32):<<1:rnd:sat`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(cmpyiwh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_cmpyiwh_PR_s1_rnd_sat(rss: i64, rt: i32) -> i32 {
    hexagon_M4_cmpyi_wh(rss, rt)
}

/// `Rd32=cmpyiwh(Rss32,Rt32*):<<1:rnd:sat`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(cmpyiwh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_cmpyiwh_PR_conj_s1_rnd_sat(rss: i64, rt: i32) -> i32 {
    hexagon_M4_cmpyi_whc(rss, rt)
}

/// `Rd32=cmpyrwh(Rss32,Rt32):<<1:rnd:sat`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(cmpyrwh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_cmpyrwh_PR_s1_rnd_sat(rss: i64, rt: i32) -> i32 {
    hexagon_M4_cmpyr_wh(rss, rt)
}

/// `Rd32=cmpyrwh(Rss32,Rt32*):<<1:rnd:sat`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(cmpyrwh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_cmpyrwh_PR_conj_s1_rnd_sat(rss: i64, rt: i32) -> i32 {
    hexagon_M4_cmpyr_whc(rss, rt)
}

/// `Rx32+=mpy(Rs32,Rt32):<<1:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpyacc_RR_s1_sat(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M4_mac_up_s1_sat(rx, rs, rt)
}

/// `Rd32=add(#u6,mpyi(Rs32,#U6))`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(0, 2)]
#[cfg_attr(test, assert_instr(add, IU6 = 0, IU6_2 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_add_mpyi_IRI<const IU6: u32, const IU6_2: u32>(rs: i32) -> i32 {
    static_assert_uimm_bits!(IU6, 6);
    static_assert_uimm_bits!(IU6_2, 6);
    hexagon_M4_mpyri_addi(IU6 as i32, rs, IU6_2 as i32)
}

/// `Rd32=add(Ru32,mpyi(Rs32,#u6))`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(add, IU6 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_add_mpyi_RRI<const IU6: u32>(ru: i32, rs: i32) -> i32 {
    static_assert_uimm_bits!(IU6, 6);
    hexagon_M4_mpyri_addr(ru, rs, IU6 as i32)
}

/// `Rd32=add(Ru32,mpyi(#u6:2,Rs32))`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(add, IU6_2 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_add_mpyi_RIR<const IU6_2: u32>(ru: i32, rs: i32) -> i32 {
    static_assert_uimm_bits!(IU6_2, 6);
    hexagon_M4_mpyri_addr_u2(ru, IU6_2 as i32, rs)
}

/// `Rd32=add(#u6,mpyi(Rs32,Rt32))`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(0)]
#[cfg_attr(test, assert_instr(add, IU6 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_add_mpyi_IRR<const IU6: u32>(rs: i32, rt: i32) -> i32 {
    static_assert_uimm_bits!(IU6, 6);
    hexagon_M4_mpyrr_addi(IU6 as i32, rs, rt)
}

/// `Ry32=add(Ru32,mpyi(Ry32,Rs32))`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(add))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_add_mpyi_RRR(ru: i32, ry: i32, rs: i32) -> i32 {
    hexagon_M4_mpyrr_addr(ru, ry, rs)
}

/// `Rx32-=mpy(Rs32,Rt32):<<1:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(mpy))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpynac_RR_s1_sat(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M4_nac_up_s1_sat(rx, rs, rt)
}

/// `Rx32|=and(Rs32,Rt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(and))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_andor_RR(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M4_or_and(rx, rs, rt)
}

/// `Rx32|=and(Rs32,~Rt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(and))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_andor_RnR(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M4_or_andn(rx, rs, rt)
}

/// `Rx32|=or(Rs32,Rt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(or))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_oror_RR(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M4_or_or(rx, rs, rt)
}

/// `Rx32|=xor(Rs32,Rt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(xor))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_xoror_RR(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M4_or_xor(rx, rs, rt)
}

/// `Rdd32=pmpyw(Rs32,Rt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(pmpyw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_pmpyw_RR(rs: i32, rt: i32) -> i64 {
    hexagon_M4_pmpyw(rs, rt)
}

/// `Rxx32^=pmpyw(Rs32,Rt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(pmpyw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_pmpywxacc_RR(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M4_pmpyw_acc(rxx, rs, rt)
}

/// `Rdd32=vpmpyh(Rs32,Rt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vpmpyh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vpmpyh_RR(rs: i32, rt: i32) -> i64 {
    hexagon_M4_vpmpyh(rs, rt)
}

/// `Rxx32^=vpmpyh(Rs32,Rt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vpmpyh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vpmpyhxacc_RR(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M4_vpmpyh_acc(rxx, rs, rt)
}

/// `Rxx32+=vrmpyweh(Rss32,Rtt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vrmpyweh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vrmpywehacc_PP(rxx: i64, rss: i64, rtt: i64) -> i64 {
    hexagon_M4_vrmpyeh_acc_s0(rxx, rss, rtt)
}

/// `Rxx32+=vrmpyweh(Rss32,Rtt32):<<1`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vrmpyweh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vrmpywehacc_PP_s1(rxx: i64, rss: i64, rtt: i64) -> i64 {
    hexagon_M4_vrmpyeh_acc_s1(rxx, rss, rtt)
}

/// `Rdd32=vrmpyweh(Rss32,Rtt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vrmpyweh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vrmpyweh_PP(rss: i64, rtt: i64) -> i64 {
    hexagon_M4_vrmpyeh_s0(rss, rtt)
}

/// `Rdd32=vrmpyweh(Rss32,Rtt32):<<1`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vrmpyweh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vrmpyweh_PP_s1(rss: i64, rtt: i64) -> i64 {
    hexagon_M4_vrmpyeh_s1(rss, rtt)
}

/// `Rxx32+=vrmpywoh(Rss32,Rtt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vrmpywoh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vrmpywohacc_PP(rxx: i64, rss: i64, rtt: i64) -> i64 {
    hexagon_M4_vrmpyoh_acc_s0(rxx, rss, rtt)
}

/// `Rxx32+=vrmpywoh(Rss32,Rtt32):<<1`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vrmpywoh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vrmpywohacc_PP_s1(rxx: i64, rss: i64, rtt: i64) -> i64 {
    hexagon_M4_vrmpyoh_acc_s1(rxx, rss, rtt)
}

/// `Rdd32=vrmpywoh(Rss32,Rtt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vrmpywoh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vrmpywoh_PP(rss: i64, rtt: i64) -> i64 {
    hexagon_M4_vrmpyoh_s0(rss, rtt)
}

/// `Rdd32=vrmpywoh(Rss32,Rtt32):<<1`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vrmpywoh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vrmpywoh_PP_s1(rss: i64, rtt: i64) -> i64 {
    hexagon_M4_vrmpyoh_s1(rss, rtt)
}

/// `Rx32^=and(Rs32,Rt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(and))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_andxacc_RR(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M4_xor_and(rx, rs, rt)
}

/// `Rx32^=and(Rs32,~Rt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(and))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_andxacc_RnR(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M4_xor_andn(rx, rs, rt)
}

/// `Rx32^=or(Rs32,Rt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(or))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_orxacc_RR(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M4_xor_or(rx, rs, rt)
}

/// `Rxx32^=xor(Rss32,Rtt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(xor))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_xorxacc_PP(rxx: i64, rss: i64, rtt: i64) -> i64 {
    hexagon_M4_xor_xacc(rxx, rss, rtt)
}

/// `Rxx32+=vdmpybsu(Rss32,Rtt32):sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vdmpybsu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vdmpybsuacc_PP_sat(rxx: i64, rss: i64, rtt: i64) -> i64 {
    hexagon_M5_vdmacbsu(rxx, rss, rtt)
}

/// `Rdd32=vdmpybsu(Rss32,Rtt32):sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vdmpybsu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vdmpybsu_PP_sat(rss: i64, rtt: i64) -> i64 {
    hexagon_M5_vdmpybsu(rss, rtt)
}

/// `Rxx32+=vmpybsu(Rs32,Rt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmpybsu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmpybsuacc_RR(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M5_vmacbsu(rxx, rs, rt)
}

/// `Rxx32+=vmpybu(Rs32,Rt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmpybu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmpybuacc_RR(rxx: i64, rs: i32, rt: i32) -> i64 {
    hexagon_M5_vmacbuu(rxx, rs, rt)
}

/// `Rdd32=vmpybsu(Rs32,Rt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmpybsu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmpybsu_RR(rs: i32, rt: i32) -> i64 {
    hexagon_M5_vmpybsu(rs, rt)
}

/// `Rdd32=vmpybu(Rs32,Rt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vmpybu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vmpybu_RR(rs: i32, rt: i32) -> i64 {
    hexagon_M5_vmpybuu(rs, rt)
}

/// `Rxx32+=vrmpybsu(Rss32,Rtt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vrmpybsu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vrmpybsuacc_PP(rxx: i64, rss: i64, rtt: i64) -> i64 {
    hexagon_M5_vrmacbsu(rxx, rss, rtt)
}

/// `Rxx32+=vrmpybu(Rss32,Rtt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vrmpybu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vrmpybuacc_PP(rxx: i64, rss: i64, rtt: i64) -> i64 {
    hexagon_M5_vrmacbuu(rxx, rss, rtt)
}

/// `Rdd32=vrmpybsu(Rss32,Rtt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vrmpybsu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vrmpybsu_PP(rss: i64, rtt: i64) -> i64 {
    hexagon_M5_vrmpybsu(rss, rtt)
}

/// `Rdd32=vrmpybu(Rss32,Rtt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vrmpybu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vrmpybu_PP(rss: i64, rtt: i64) -> i64 {
    hexagon_M5_vrmpybuu(rss, rtt)
}

/// `Rd32=addasl(Rt32,Rs32,#u3)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(addasl, IU3 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_addasl_RRI<const IU3: u32>(rt: i32, rs: i32) -> i32 {
    static_assert_uimm_bits!(IU3, 3);
    hexagon_S2_addasl_rrri(rt, rs, IU3 as i32)
}

/// `Rdd32=asl(Rss32,#u6)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(asl, IU6 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_asl_PI<const IU6: u32>(rss: i64) -> i64 {
    static_assert_uimm_bits!(IU6, 6);
    hexagon_S2_asl_i_p(rss, IU6 as i32)
}

/// `Rxx32+=asl(Rss32,#u6)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(asl, IU6 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_aslacc_PI<const IU6: u32>(rxx: i64, rss: i64) -> i64 {
    static_assert_uimm_bits!(IU6, 6);
    hexagon_S2_asl_i_p_acc(rxx, rss, IU6 as i32)
}

/// `Rxx32&=asl(Rss32,#u6)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(asl, IU6 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_asland_PI<const IU6: u32>(rxx: i64, rss: i64) -> i64 {
    static_assert_uimm_bits!(IU6, 6);
    hexagon_S2_asl_i_p_and(rxx, rss, IU6 as i32)
}

/// `Rxx32-=asl(Rss32,#u6)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(asl, IU6 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_aslnac_PI<const IU6: u32>(rxx: i64, rss: i64) -> i64 {
    static_assert_uimm_bits!(IU6, 6);
    hexagon_S2_asl_i_p_nac(rxx, rss, IU6 as i32)
}

/// `Rxx32|=asl(Rss32,#u6)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(asl, IU6 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_aslor_PI<const IU6: u32>(rxx: i64, rss: i64) -> i64 {
    static_assert_uimm_bits!(IU6, 6);
    hexagon_S2_asl_i_p_or(rxx, rss, IU6 as i32)
}

/// `Rxx32^=asl(Rss32,#u6)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(asl, IU6 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_aslxacc_PI<const IU6: u32>(rxx: i64, rss: i64) -> i64 {
    static_assert_uimm_bits!(IU6, 6);
    hexagon_S2_asl_i_p_xacc(rxx, rss, IU6 as i32)
}

/// `Rd32=asl(Rs32,#u5)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(asl, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_asl_RI<const IU5: u32>(rs: i32) -> i32 {
    static_assert_uimm_bits!(IU5, 5);
    hexagon_S2_asl_i_r(rs, IU5 as i32)
}

/// `Rx32+=asl(Rs32,#u5)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(asl, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_aslacc_RI<const IU5: u32>(rx: i32, rs: i32) -> i32 {
    static_assert_uimm_bits!(IU5, 5);
    hexagon_S2_asl_i_r_acc(rx, rs, IU5 as i32)
}

/// `Rx32&=asl(Rs32,#u5)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(asl, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_asland_RI<const IU5: u32>(rx: i32, rs: i32) -> i32 {
    static_assert_uimm_bits!(IU5, 5);
    hexagon_S2_asl_i_r_and(rx, rs, IU5 as i32)
}

/// `Rx32-=asl(Rs32,#u5)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(asl, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_aslnac_RI<const IU5: u32>(rx: i32, rs: i32) -> i32 {
    static_assert_uimm_bits!(IU5, 5);
    hexagon_S2_asl_i_r_nac(rx, rs, IU5 as i32)
}

/// `Rx32|=asl(Rs32,#u5)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(asl, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_aslor_RI<const IU5: u32>(rx: i32, rs: i32) -> i32 {
    static_assert_uimm_bits!(IU5, 5);
    hexagon_S2_asl_i_r_or(rx, rs, IU5 as i32)
}

/// `Rd32=asl(Rs32,#u5):sat`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(asl, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_asl_RI_sat<const IU5: u32>(rs: i32) -> i32 {
    static_assert_uimm_bits!(IU5, 5);
    hexagon_S2_asl_i_r_sat(rs, IU5 as i32)
}

/// `Rx32^=asl(Rs32,#u5)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(asl, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_aslxacc_RI<const IU5: u32>(rx: i32, rs: i32) -> i32 {
    static_assert_uimm_bits!(IU5, 5);
    hexagon_S2_asl_i_r_xacc(rx, rs, IU5 as i32)
}

/// `Rdd32=vaslh(Rss32,#u4)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(vaslh, IU4 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vaslh_PI<const IU4: u32>(rss: i64) -> i64 {
    static_assert_uimm_bits!(IU4, 4);
    hexagon_S2_asl_i_vh(rss, IU4 as i32)
}

/// `Rdd32=vaslw(Rss32,#u5)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(vaslw, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vaslw_PI<const IU5: u32>(rss: i64) -> i64 {
    static_assert_uimm_bits!(IU5, 5);
    hexagon_S2_asl_i_vw(rss, IU5 as i32)
}

/// `Rdd32=asl(Rss32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(asl))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_asl_PR(rss: i64, rt: i32) -> i64 {
    hexagon_S2_asl_r_p(rss, rt)
}

/// `Rxx32+=asl(Rss32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(asl))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_aslacc_PR(rxx: i64, rss: i64, rt: i32) -> i64 {
    hexagon_S2_asl_r_p_acc(rxx, rss, rt)
}

/// `Rxx32&=asl(Rss32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(asl))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_asland_PR(rxx: i64, rss: i64, rt: i32) -> i64 {
    hexagon_S2_asl_r_p_and(rxx, rss, rt)
}

/// `Rxx32-=asl(Rss32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(asl))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_aslnac_PR(rxx: i64, rss: i64, rt: i32) -> i64 {
    hexagon_S2_asl_r_p_nac(rxx, rss, rt)
}

/// `Rxx32|=asl(Rss32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(asl))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_aslor_PR(rxx: i64, rss: i64, rt: i32) -> i64 {
    hexagon_S2_asl_r_p_or(rxx, rss, rt)
}

/// `Rxx32^=asl(Rss32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(asl))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_aslxacc_PR(rxx: i64, rss: i64, rt: i32) -> i64 {
    hexagon_S2_asl_r_p_xor(rxx, rss, rt)
}

/// `Rd32=asl(Rs32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(asl))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_asl_RR(rs: i32, rt: i32) -> i32 {
    hexagon_S2_asl_r_r(rs, rt)
}

/// `Rx32+=asl(Rs32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(asl))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_aslacc_RR(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_S2_asl_r_r_acc(rx, rs, rt)
}

/// `Rx32&=asl(Rs32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(asl))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_asland_RR(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_S2_asl_r_r_and(rx, rs, rt)
}

/// `Rx32-=asl(Rs32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(asl))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_aslnac_RR(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_S2_asl_r_r_nac(rx, rs, rt)
}

/// `Rx32|=asl(Rs32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(asl))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_aslor_RR(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_S2_asl_r_r_or(rx, rs, rt)
}

/// `Rd32=asl(Rs32,Rt32):sat`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(asl))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_asl_RR_sat(rs: i32, rt: i32) -> i32 {
    hexagon_S2_asl_r_r_sat(rs, rt)
}

/// `Rdd32=vaslh(Rss32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vaslh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vaslh_PR(rss: i64, rt: i32) -> i64 {
    hexagon_S2_asl_r_vh(rss, rt)
}

/// `Rdd32=vaslw(Rss32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vaslw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vaslw_PR(rss: i64, rt: i32) -> i64 {
    hexagon_S2_asl_r_vw(rss, rt)
}

/// `Rdd32=asr(Rss32,#u6)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(asr, IU6 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_asr_PI<const IU6: u32>(rss: i64) -> i64 {
    static_assert_uimm_bits!(IU6, 6);
    hexagon_S2_asr_i_p(rss, IU6 as i32)
}

/// `Rxx32+=asr(Rss32,#u6)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(asr, IU6 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_asracc_PI<const IU6: u32>(rxx: i64, rss: i64) -> i64 {
    static_assert_uimm_bits!(IU6, 6);
    hexagon_S2_asr_i_p_acc(rxx, rss, IU6 as i32)
}

/// `Rxx32&=asr(Rss32,#u6)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(asr, IU6 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_asrand_PI<const IU6: u32>(rxx: i64, rss: i64) -> i64 {
    static_assert_uimm_bits!(IU6, 6);
    hexagon_S2_asr_i_p_and(rxx, rss, IU6 as i32)
}

/// `Rxx32-=asr(Rss32,#u6)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(asr, IU6 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_asrnac_PI<const IU6: u32>(rxx: i64, rss: i64) -> i64 {
    static_assert_uimm_bits!(IU6, 6);
    hexagon_S2_asr_i_p_nac(rxx, rss, IU6 as i32)
}

/// `Rxx32|=asr(Rss32,#u6)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(asr, IU6 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_asror_PI<const IU6: u32>(rxx: i64, rss: i64) -> i64 {
    static_assert_uimm_bits!(IU6, 6);
    hexagon_S2_asr_i_p_or(rxx, rss, IU6 as i32)
}

/// `Rdd32=asr(Rss32,#u6):rnd`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(asr, IU6 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_asr_PI_rnd<const IU6: u32>(rss: i64) -> i64 {
    static_assert_uimm_bits!(IU6, 6);
    hexagon_S2_asr_i_p_rnd(rss, IU6 as i32)
}

/// `Rdd32=asrrnd(Rss32,#u6)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT0123
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(asrrnd, IU6 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_asrrnd_PI<const IU6: u32>(rss: i64) -> i64 {
    static_assert_uimm_bits!(IU6, 6);
    hexagon_S2_asr_i_p_rnd_goodsyntax(rss, IU6 as i32)
}

/// `Rd32=asr(Rs32,#u5)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(asr, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_asr_RI<const IU5: u32>(rs: i32) -> i32 {
    static_assert_uimm_bits!(IU5, 5);
    hexagon_S2_asr_i_r(rs, IU5 as i32)
}

/// `Rx32+=asr(Rs32,#u5)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(asr, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_asracc_RI<const IU5: u32>(rx: i32, rs: i32) -> i32 {
    static_assert_uimm_bits!(IU5, 5);
    hexagon_S2_asr_i_r_acc(rx, rs, IU5 as i32)
}

/// `Rx32&=asr(Rs32,#u5)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(asr, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_asrand_RI<const IU5: u32>(rx: i32, rs: i32) -> i32 {
    static_assert_uimm_bits!(IU5, 5);
    hexagon_S2_asr_i_r_and(rx, rs, IU5 as i32)
}

/// `Rx32-=asr(Rs32,#u5)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(asr, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_asrnac_RI<const IU5: u32>(rx: i32, rs: i32) -> i32 {
    static_assert_uimm_bits!(IU5, 5);
    hexagon_S2_asr_i_r_nac(rx, rs, IU5 as i32)
}

/// `Rx32|=asr(Rs32,#u5)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(asr, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_asror_RI<const IU5: u32>(rx: i32, rs: i32) -> i32 {
    static_assert_uimm_bits!(IU5, 5);
    hexagon_S2_asr_i_r_or(rx, rs, IU5 as i32)
}

/// `Rd32=asr(Rs32,#u5):rnd`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(asr, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_asr_RI_rnd<const IU5: u32>(rs: i32) -> i32 {
    static_assert_uimm_bits!(IU5, 5);
    hexagon_S2_asr_i_r_rnd(rs, IU5 as i32)
}

/// `Rd32=asrrnd(Rs32,#u5)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT0123
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(asrrnd, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_asrrnd_RI<const IU5: u32>(rs: i32) -> i32 {
    static_assert_uimm_bits!(IU5, 5);
    hexagon_S2_asr_i_r_rnd_goodsyntax(rs, IU5 as i32)
}

/// `Rd32=vasrw(Rss32,#u5)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(vasrw, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_vasrw_PI<const IU5: u32>(rss: i64) -> i32 {
    static_assert_uimm_bits!(IU5, 5);
    hexagon_S2_asr_i_svw_trun(rss, IU5 as i32)
}

/// `Rdd32=vasrh(Rss32,#u4)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(vasrh, IU4 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vasrh_PI<const IU4: u32>(rss: i64) -> i64 {
    static_assert_uimm_bits!(IU4, 4);
    hexagon_S2_asr_i_vh(rss, IU4 as i32)
}

/// `Rdd32=vasrw(Rss32,#u5)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(vasrw, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vasrw_PI<const IU5: u32>(rss: i64) -> i64 {
    static_assert_uimm_bits!(IU5, 5);
    hexagon_S2_asr_i_vw(rss, IU5 as i32)
}

/// `Rdd32=asr(Rss32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(asr))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_asr_PR(rss: i64, rt: i32) -> i64 {
    hexagon_S2_asr_r_p(rss, rt)
}

/// `Rxx32+=asr(Rss32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(asr))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_asracc_PR(rxx: i64, rss: i64, rt: i32) -> i64 {
    hexagon_S2_asr_r_p_acc(rxx, rss, rt)
}

/// `Rxx32&=asr(Rss32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(asr))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_asrand_PR(rxx: i64, rss: i64, rt: i32) -> i64 {
    hexagon_S2_asr_r_p_and(rxx, rss, rt)
}

/// `Rxx32-=asr(Rss32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(asr))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_asrnac_PR(rxx: i64, rss: i64, rt: i32) -> i64 {
    hexagon_S2_asr_r_p_nac(rxx, rss, rt)
}

/// `Rxx32|=asr(Rss32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(asr))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_asror_PR(rxx: i64, rss: i64, rt: i32) -> i64 {
    hexagon_S2_asr_r_p_or(rxx, rss, rt)
}

/// `Rxx32^=asr(Rss32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(asr))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_asrxacc_PR(rxx: i64, rss: i64, rt: i32) -> i64 {
    hexagon_S2_asr_r_p_xor(rxx, rss, rt)
}

/// `Rd32=asr(Rs32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(asr))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_asr_RR(rs: i32, rt: i32) -> i32 {
    hexagon_S2_asr_r_r(rs, rt)
}

/// `Rx32+=asr(Rs32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(asr))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_asracc_RR(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_S2_asr_r_r_acc(rx, rs, rt)
}

/// `Rx32&=asr(Rs32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(asr))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_asrand_RR(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_S2_asr_r_r_and(rx, rs, rt)
}

/// `Rx32-=asr(Rs32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(asr))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_asrnac_RR(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_S2_asr_r_r_nac(rx, rs, rt)
}

/// `Rx32|=asr(Rs32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(asr))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_asror_RR(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_S2_asr_r_r_or(rx, rs, rt)
}

/// `Rd32=asr(Rs32,Rt32):sat`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(asr))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_asr_RR_sat(rs: i32, rt: i32) -> i32 {
    hexagon_S2_asr_r_r_sat(rs, rt)
}

/// `Rd32=vasrw(Rss32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vasrw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_vasrw_PR(rss: i64, rt: i32) -> i32 {
    hexagon_S2_asr_r_svw_trun(rss, rt)
}

/// `Rdd32=vasrh(Rss32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vasrh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vasrh_PR(rss: i64, rt: i32) -> i64 {
    hexagon_S2_asr_r_vh(rss, rt)
}

/// `Rdd32=vasrw(Rss32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vasrw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vasrw_PR(rss: i64, rt: i32) -> i64 {
    hexagon_S2_asr_r_vw(rss, rt)
}

/// `Rd32=brev(Rs32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(brev))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_brev_R(rs: i32) -> i32 {
    hexagon_S2_brev(rs)
}

/// `Rdd32=brev(Rss32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(brev))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_brev_P(rss: i64) -> i64 {
    hexagon_S2_brevp(rss)
}

/// `Rd32=cl0(Rs32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(cl0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_cl0_R(rs: i32) -> i32 {
    hexagon_S2_cl0(rs)
}

/// `Rd32=cl0(Rss32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(cl0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_cl0_P(rss: i64) -> i32 {
    hexagon_S2_cl0p(rss)
}

/// `Rd32=cl1(Rs32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(cl1))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_cl1_R(rs: i32) -> i32 {
    hexagon_S2_cl1(rs)
}

/// `Rd32=cl1(Rss32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(cl1))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_cl1_P(rss: i64) -> i32 {
    hexagon_S2_cl1p(rss)
}

/// `Rd32=clb(Rs32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(clb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_clb_R(rs: i32) -> i32 {
    hexagon_S2_clb(rs)
}

/// `Rd32=normamt(Rs32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(normamt))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_normamt_R(rs: i32) -> i32 {
    hexagon_S2_clbnorm(rs)
}

/// `Rd32=clb(Rss32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(clb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_clb_P(rss: i64) -> i32 {
    hexagon_S2_clbp(rss)
}

/// `Rd32=clrbit(Rs32,#u5)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(clrbit, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_clrbit_RI<const IU5: u32>(rs: i32) -> i32 {
    static_assert_uimm_bits!(IU5, 5);
    hexagon_S2_clrbit_i(rs, IU5 as i32)
}

/// `Rd32=clrbit(Rs32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(clrbit))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_clrbit_RR(rs: i32, rt: i32) -> i32 {
    hexagon_S2_clrbit_r(rs, rt)
}

/// `Rd32=ct0(Rs32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(ct0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_ct0_R(rs: i32) -> i32 {
    hexagon_S2_ct0(rs)
}

/// `Rd32=ct0(Rss32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(ct0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_ct0_P(rss: i64) -> i32 {
    hexagon_S2_ct0p(rss)
}

/// `Rd32=ct1(Rs32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(ct1))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_ct1_R(rs: i32) -> i32 {
    hexagon_S2_ct1(rs)
}

/// `Rd32=ct1(Rss32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(ct1))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_ct1_P(rss: i64) -> i32 {
    hexagon_S2_ct1p(rss)
}

/// `Rdd32=deinterleave(Rss32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(deinterleave))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_deinterleave_P(rss: i64) -> i64 {
    hexagon_S2_deinterleave(rss)
}

/// `Rd32=extractu(Rs32,#u5,#U5)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(1, 2)]
#[cfg_attr(test, assert_instr(extractu, IU5 = 0, IU5_2 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_extractu_RII<const IU5: u32, const IU5_2: u32>(rs: i32) -> i32 {
    static_assert_uimm_bits!(IU5, 5);
    static_assert_uimm_bits!(IU5_2, 5);
    hexagon_S2_extractu(rs, IU5 as i32, IU5_2 as i32)
}

/// `Rd32=extractu(Rs32,Rtt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(extractu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_extractu_RP(rs: i32, rtt: i64) -> i32 {
    hexagon_S2_extractu_rp(rs, rtt)
}

/// `Rdd32=extractu(Rss32,#u6,#U6)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(1, 2)]
#[cfg_attr(test, assert_instr(extractu, IU6 = 0, IU6_2 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_extractu_PII<const IU6: u32, const IU6_2: u32>(rss: i64) -> i64 {
    static_assert_uimm_bits!(IU6, 6);
    static_assert_uimm_bits!(IU6_2, 6);
    hexagon_S2_extractup(rss, IU6 as i32, IU6_2 as i32)
}

/// `Rdd32=extractu(Rss32,Rtt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(extractu))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_extractu_PP(rss: i64, rtt: i64) -> i64 {
    hexagon_S2_extractup_rp(rss, rtt)
}

/// `Rx32=insert(Rs32,#u5,#U5)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(2, 3)]
#[cfg_attr(test, assert_instr(insert, IU5 = 0, IU5_2 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_insert_RII<const IU5: u32, const IU5_2: u32>(rx: i32, rs: i32) -> i32 {
    static_assert_uimm_bits!(IU5, 5);
    static_assert_uimm_bits!(IU5_2, 5);
    hexagon_S2_insert(rx, rs, IU5 as i32, IU5_2 as i32)
}

/// `Rx32=insert(Rs32,Rtt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(insert))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_insert_RP(rx: i32, rs: i32, rtt: i64) -> i32 {
    hexagon_S2_insert_rp(rx, rs, rtt)
}

/// `Rxx32=insert(Rss32,#u6,#U6)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(2, 3)]
#[cfg_attr(test, assert_instr(insert, IU6 = 0, IU6_2 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_insert_PII<const IU6: u32, const IU6_2: u32>(rxx: i64, rss: i64) -> i64 {
    static_assert_uimm_bits!(IU6, 6);
    static_assert_uimm_bits!(IU6_2, 6);
    hexagon_S2_insertp(rxx, rss, IU6 as i32, IU6_2 as i32)
}

/// `Rxx32=insert(Rss32,Rtt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(insert))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_insert_PP(rxx: i64, rss: i64, rtt: i64) -> i64 {
    hexagon_S2_insertp_rp(rxx, rss, rtt)
}

/// `Rdd32=interleave(Rss32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(interleave))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_interleave_P(rss: i64) -> i64 {
    hexagon_S2_interleave(rss)
}

/// `Rdd32=lfs(Rss32,Rtt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(lfs))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_lfs_PP(rss: i64, rtt: i64) -> i64 {
    hexagon_S2_lfsp(rss, rtt)
}

/// `Rdd32=lsl(Rss32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(lsl))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_lsl_PR(rss: i64, rt: i32) -> i64 {
    hexagon_S2_lsl_r_p(rss, rt)
}

/// `Rxx32+=lsl(Rss32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(lsl))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_lslacc_PR(rxx: i64, rss: i64, rt: i32) -> i64 {
    hexagon_S2_lsl_r_p_acc(rxx, rss, rt)
}

/// `Rxx32&=lsl(Rss32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(lsl))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_lsland_PR(rxx: i64, rss: i64, rt: i32) -> i64 {
    hexagon_S2_lsl_r_p_and(rxx, rss, rt)
}

/// `Rxx32-=lsl(Rss32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(lsl))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_lslnac_PR(rxx: i64, rss: i64, rt: i32) -> i64 {
    hexagon_S2_lsl_r_p_nac(rxx, rss, rt)
}

/// `Rxx32|=lsl(Rss32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(lsl))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_lslor_PR(rxx: i64, rss: i64, rt: i32) -> i64 {
    hexagon_S2_lsl_r_p_or(rxx, rss, rt)
}

/// `Rxx32^=lsl(Rss32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(lsl))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_lslxacc_PR(rxx: i64, rss: i64, rt: i32) -> i64 {
    hexagon_S2_lsl_r_p_xor(rxx, rss, rt)
}

/// `Rd32=lsl(Rs32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(lsl))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_lsl_RR(rs: i32, rt: i32) -> i32 {
    hexagon_S2_lsl_r_r(rs, rt)
}

/// `Rx32+=lsl(Rs32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(lsl))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_lslacc_RR(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_S2_lsl_r_r_acc(rx, rs, rt)
}

/// `Rx32&=lsl(Rs32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(lsl))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_lsland_RR(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_S2_lsl_r_r_and(rx, rs, rt)
}

/// `Rx32-=lsl(Rs32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(lsl))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_lslnac_RR(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_S2_lsl_r_r_nac(rx, rs, rt)
}

/// `Rx32|=lsl(Rs32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(lsl))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_lslor_RR(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_S2_lsl_r_r_or(rx, rs, rt)
}

/// `Rdd32=vlslh(Rss32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vlslh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vlslh_PR(rss: i64, rt: i32) -> i64 {
    hexagon_S2_lsl_r_vh(rss, rt)
}

/// `Rdd32=vlslw(Rss32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vlslw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vlslw_PR(rss: i64, rt: i32) -> i64 {
    hexagon_S2_lsl_r_vw(rss, rt)
}

/// `Rdd32=lsr(Rss32,#u6)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(lsr, IU6 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_lsr_PI<const IU6: u32>(rss: i64) -> i64 {
    static_assert_uimm_bits!(IU6, 6);
    hexagon_S2_lsr_i_p(rss, IU6 as i32)
}

/// `Rxx32+=lsr(Rss32,#u6)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(lsr, IU6 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_lsracc_PI<const IU6: u32>(rxx: i64, rss: i64) -> i64 {
    static_assert_uimm_bits!(IU6, 6);
    hexagon_S2_lsr_i_p_acc(rxx, rss, IU6 as i32)
}

/// `Rxx32&=lsr(Rss32,#u6)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(lsr, IU6 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_lsrand_PI<const IU6: u32>(rxx: i64, rss: i64) -> i64 {
    static_assert_uimm_bits!(IU6, 6);
    hexagon_S2_lsr_i_p_and(rxx, rss, IU6 as i32)
}

/// `Rxx32-=lsr(Rss32,#u6)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(lsr, IU6 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_lsrnac_PI<const IU6: u32>(rxx: i64, rss: i64) -> i64 {
    static_assert_uimm_bits!(IU6, 6);
    hexagon_S2_lsr_i_p_nac(rxx, rss, IU6 as i32)
}

/// `Rxx32|=lsr(Rss32,#u6)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(lsr, IU6 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_lsror_PI<const IU6: u32>(rxx: i64, rss: i64) -> i64 {
    static_assert_uimm_bits!(IU6, 6);
    hexagon_S2_lsr_i_p_or(rxx, rss, IU6 as i32)
}

/// `Rxx32^=lsr(Rss32,#u6)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(lsr, IU6 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_lsrxacc_PI<const IU6: u32>(rxx: i64, rss: i64) -> i64 {
    static_assert_uimm_bits!(IU6, 6);
    hexagon_S2_lsr_i_p_xacc(rxx, rss, IU6 as i32)
}

/// `Rd32=lsr(Rs32,#u5)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(lsr, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_lsr_RI<const IU5: u32>(rs: i32) -> i32 {
    static_assert_uimm_bits!(IU5, 5);
    hexagon_S2_lsr_i_r(rs, IU5 as i32)
}

/// `Rx32+=lsr(Rs32,#u5)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(lsr, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_lsracc_RI<const IU5: u32>(rx: i32, rs: i32) -> i32 {
    static_assert_uimm_bits!(IU5, 5);
    hexagon_S2_lsr_i_r_acc(rx, rs, IU5 as i32)
}

/// `Rx32&=lsr(Rs32,#u5)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(lsr, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_lsrand_RI<const IU5: u32>(rx: i32, rs: i32) -> i32 {
    static_assert_uimm_bits!(IU5, 5);
    hexagon_S2_lsr_i_r_and(rx, rs, IU5 as i32)
}

/// `Rx32-=lsr(Rs32,#u5)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(lsr, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_lsrnac_RI<const IU5: u32>(rx: i32, rs: i32) -> i32 {
    static_assert_uimm_bits!(IU5, 5);
    hexagon_S2_lsr_i_r_nac(rx, rs, IU5 as i32)
}

/// `Rx32|=lsr(Rs32,#u5)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(lsr, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_lsror_RI<const IU5: u32>(rx: i32, rs: i32) -> i32 {
    static_assert_uimm_bits!(IU5, 5);
    hexagon_S2_lsr_i_r_or(rx, rs, IU5 as i32)
}

/// `Rx32^=lsr(Rs32,#u5)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(lsr, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_lsrxacc_RI<const IU5: u32>(rx: i32, rs: i32) -> i32 {
    static_assert_uimm_bits!(IU5, 5);
    hexagon_S2_lsr_i_r_xacc(rx, rs, IU5 as i32)
}

/// `Rdd32=vlsrh(Rss32,#u4)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(vlsrh, IU4 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vlsrh_PI<const IU4: u32>(rss: i64) -> i64 {
    static_assert_uimm_bits!(IU4, 4);
    hexagon_S2_lsr_i_vh(rss, IU4 as i32)
}

/// `Rdd32=vlsrw(Rss32,#u5)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(vlsrw, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vlsrw_PI<const IU5: u32>(rss: i64) -> i64 {
    static_assert_uimm_bits!(IU5, 5);
    hexagon_S2_lsr_i_vw(rss, IU5 as i32)
}

/// `Rdd32=lsr(Rss32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(lsr))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_lsr_PR(rss: i64, rt: i32) -> i64 {
    hexagon_S2_lsr_r_p(rss, rt)
}

/// `Rxx32+=lsr(Rss32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(lsr))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_lsracc_PR(rxx: i64, rss: i64, rt: i32) -> i64 {
    hexagon_S2_lsr_r_p_acc(rxx, rss, rt)
}

/// `Rxx32&=lsr(Rss32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(lsr))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_lsrand_PR(rxx: i64, rss: i64, rt: i32) -> i64 {
    hexagon_S2_lsr_r_p_and(rxx, rss, rt)
}

/// `Rxx32-=lsr(Rss32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(lsr))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_lsrnac_PR(rxx: i64, rss: i64, rt: i32) -> i64 {
    hexagon_S2_lsr_r_p_nac(rxx, rss, rt)
}

/// `Rxx32|=lsr(Rss32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(lsr))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_lsror_PR(rxx: i64, rss: i64, rt: i32) -> i64 {
    hexagon_S2_lsr_r_p_or(rxx, rss, rt)
}

/// `Rxx32^=lsr(Rss32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(lsr))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_lsrxacc_PR(rxx: i64, rss: i64, rt: i32) -> i64 {
    hexagon_S2_lsr_r_p_xor(rxx, rss, rt)
}

/// `Rd32=lsr(Rs32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(lsr))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_lsr_RR(rs: i32, rt: i32) -> i32 {
    hexagon_S2_lsr_r_r(rs, rt)
}

/// `Rx32+=lsr(Rs32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(lsr))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_lsracc_RR(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_S2_lsr_r_r_acc(rx, rs, rt)
}

/// `Rx32&=lsr(Rs32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(lsr))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_lsrand_RR(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_S2_lsr_r_r_and(rx, rs, rt)
}

/// `Rx32-=lsr(Rs32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(lsr))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_lsrnac_RR(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_S2_lsr_r_r_nac(rx, rs, rt)
}

/// `Rx32|=lsr(Rs32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(lsr))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_lsror_RR(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_S2_lsr_r_r_or(rx, rs, rt)
}

/// `Rdd32=vlsrh(Rss32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vlsrh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vlsrh_PR(rss: i64, rt: i32) -> i64 {
    hexagon_S2_lsr_r_vh(rss, rt)
}

/// `Rdd32=vlsrw(Rss32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vlsrw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vlsrw_PR(rss: i64, rt: i32) -> i64 {
    hexagon_S2_lsr_r_vw(rss, rt)
}

/// `Rdd32=packhl(Rs32,Rt32)`
///
/// Instruction Type: ALU32_3op
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(test, assert_instr(packhl))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_packhl_RR(rs: i32, rt: i32) -> i64 {
    hexagon_S2_packhl(rs, rt)
}

/// `Rd32=parity(Rss32,Rtt32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(parity))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_parity_PP(rss: i64, rtt: i64) -> i32 {
    hexagon_S2_parityp(rss, rtt)
}

/// `Rd32=setbit(Rs32,#u5)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(setbit, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_setbit_RI<const IU5: u32>(rs: i32) -> i32 {
    static_assert_uimm_bits!(IU5, 5);
    hexagon_S2_setbit_i(rs, IU5 as i32)
}

/// `Rd32=setbit(Rs32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(setbit))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_setbit_RR(rs: i32, rt: i32) -> i32 {
    hexagon_S2_setbit_r(rs, rt)
}

/// `Rdd32=shuffeb(Rss32,Rtt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(shuffeb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_shuffeb_PP(rss: i64, rtt: i64) -> i64 {
    hexagon_S2_shuffeb(rss, rtt)
}

/// `Rdd32=shuffeh(Rss32,Rtt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(shuffeh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_shuffeh_PP(rss: i64, rtt: i64) -> i64 {
    hexagon_S2_shuffeh(rss, rtt)
}

/// `Rdd32=shuffob(Rtt32,Rss32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(shuffob))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_shuffob_PP(rtt: i64, rss: i64) -> i64 {
    hexagon_S2_shuffob(rtt, rss)
}

/// `Rdd32=shuffoh(Rtt32,Rss32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(shuffoh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_shuffoh_PP(rtt: i64, rss: i64) -> i64 {
    hexagon_S2_shuffoh(rtt, rss)
}

/// `Rd32=vsathb(Rs32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vsathb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_vsathb_R(rs: i32) -> i32 {
    hexagon_S2_svsathb(rs)
}

/// `Rd32=vsathub(Rs32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vsathub))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_vsathub_R(rs: i32) -> i32 {
    hexagon_S2_svsathub(rs)
}

/// `Rx32=tableidxb(Rs32,#u4,#U5)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT0123
#[inline(always)]
#[rustc_legacy_const_generics(2, 3)]
#[cfg_attr(test, assert_instr(tableidxb, IU4 = 0, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_tableidxb_RII<const IU4: u32, const IU5: u32>(rx: i32, rs: i32) -> i32 {
    static_assert_uimm_bits!(IU4, 4);
    static_assert_uimm_bits!(IU5, 5);
    hexagon_S2_tableidxb_goodsyntax(rx, rs, IU4 as i32, IU5 as i32)
}

/// `Rx32=tableidxd(Rs32,#u4,#U5)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT0123
#[inline(always)]
#[rustc_legacy_const_generics(2, 3)]
#[cfg_attr(test, assert_instr(tableidxd, IU4 = 0, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_tableidxd_RII<const IU4: u32, const IU5: u32>(rx: i32, rs: i32) -> i32 {
    static_assert_uimm_bits!(IU4, 4);
    static_assert_uimm_bits!(IU5, 5);
    hexagon_S2_tableidxd_goodsyntax(rx, rs, IU4 as i32, IU5 as i32)
}

/// `Rx32=tableidxh(Rs32,#u4,#U5)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT0123
#[inline(always)]
#[rustc_legacy_const_generics(2, 3)]
#[cfg_attr(test, assert_instr(tableidxh, IU4 = 0, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_tableidxh_RII<const IU4: u32, const IU5: u32>(rx: i32, rs: i32) -> i32 {
    static_assert_uimm_bits!(IU4, 4);
    static_assert_uimm_bits!(IU5, 5);
    hexagon_S2_tableidxh_goodsyntax(rx, rs, IU4 as i32, IU5 as i32)
}

/// `Rx32=tableidxw(Rs32,#u4,#U5)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT0123
#[inline(always)]
#[rustc_legacy_const_generics(2, 3)]
#[cfg_attr(test, assert_instr(tableidxw, IU4 = 0, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_tableidxw_RII<const IU4: u32, const IU5: u32>(rx: i32, rs: i32) -> i32 {
    static_assert_uimm_bits!(IU4, 4);
    static_assert_uimm_bits!(IU5, 5);
    hexagon_S2_tableidxw_goodsyntax(rx, rs, IU4 as i32, IU5 as i32)
}

/// `Rd32=togglebit(Rs32,#u5)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(togglebit, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_togglebit_RI<const IU5: u32>(rs: i32) -> i32 {
    static_assert_uimm_bits!(IU5, 5);
    hexagon_S2_togglebit_i(rs, IU5 as i32)
}

/// `Rd32=togglebit(Rs32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(togglebit))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_togglebit_RR(rs: i32, rt: i32) -> i32 {
    hexagon_S2_togglebit_r(rs, rt)
}

/// `Pd4=tstbit(Rs32,#u5)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(tstbit, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_tstbit_RI<const IU5: u32>(rs: i32) -> i32 {
    static_assert_uimm_bits!(IU5, 5);
    hexagon_S2_tstbit_i(rs, IU5 as i32)
}

/// `Pd4=tstbit(Rs32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(tstbit))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_tstbit_RR(rs: i32, rt: i32) -> i32 {
    hexagon_S2_tstbit_r(rs, rt)
}

/// `Rdd32=valignb(Rtt32,Rss32,#u3)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(valignb, IU3 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_valignb_PPI<const IU3: u32>(rtt: i64, rss: i64) -> i64 {
    static_assert_uimm_bits!(IU3, 3);
    hexagon_S2_valignib(rtt, rss, IU3 as i32)
}

/// `Rdd32=valignb(Rtt32,Rss32,Pu4)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(valignb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_valignb_PPp(rtt: i64, rss: i64, pu: i32) -> i64 {
    hexagon_S2_valignrb(rtt, rss, pu)
}

/// `Rdd32=vcnegh(Rss32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vcnegh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vcnegh_PR(rss: i64, rt: i32) -> i64 {
    hexagon_S2_vcnegh(rss, rt)
}

/// `Rdd32=vcrotate(Rss32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vcrotate))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vcrotate_PR(rss: i64, rt: i32) -> i64 {
    hexagon_S2_vcrotate(rss, rt)
}

/// `Rxx32+=vrcnegh(Rss32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vrcnegh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vrcneghacc_PR(rxx: i64, rss: i64, rt: i32) -> i64 {
    hexagon_S2_vrcnegh(rxx, rss, rt)
}

/// `Rd32=vrndwh(Rss32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vrndwh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_vrndwh_P(rss: i64) -> i32 {
    hexagon_S2_vrndpackwh(rss)
}

/// `Rd32=vrndwh(Rss32):sat`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vrndwh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_vrndwh_P_sat(rss: i64) -> i32 {
    hexagon_S2_vrndpackwhs(rss)
}

/// `Rd32=vsathb(Rss32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vsathb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_vsathb_P(rss: i64) -> i32 {
    hexagon_S2_vsathb(rss)
}

/// `Rdd32=vsathb(Rss32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vsathb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vsathb_P(rss: i64) -> i64 {
    hexagon_S2_vsathb_nopack(rss)
}

/// `Rd32=vsathub(Rss32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vsathub))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_vsathub_P(rss: i64) -> i32 {
    hexagon_S2_vsathub(rss)
}

/// `Rdd32=vsathub(Rss32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vsathub))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vsathub_P(rss: i64) -> i64 {
    hexagon_S2_vsathub_nopack(rss)
}

/// `Rd32=vsatwh(Rss32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vsatwh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_vsatwh_P(rss: i64) -> i32 {
    hexagon_S2_vsatwh(rss)
}

/// `Rdd32=vsatwh(Rss32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vsatwh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vsatwh_P(rss: i64) -> i64 {
    hexagon_S2_vsatwh_nopack(rss)
}

/// `Rd32=vsatwuh(Rss32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vsatwuh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_vsatwuh_P(rss: i64) -> i32 {
    hexagon_S2_vsatwuh(rss)
}

/// `Rdd32=vsatwuh(Rss32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vsatwuh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vsatwuh_P(rss: i64) -> i64 {
    hexagon_S2_vsatwuh_nopack(rss)
}

/// `Rd32=vsplatb(Rs32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vsplatb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_vsplatb_R(rs: i32) -> i32 {
    hexagon_S2_vsplatrb(rs)
}

/// `Rdd32=vsplath(Rs32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vsplath))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vsplath_R(rs: i32) -> i64 {
    hexagon_S2_vsplatrh(rs)
}

/// `Rdd32=vspliceb(Rss32,Rtt32,#u3)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(vspliceb, IU3 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vspliceb_PPI<const IU3: u32>(rss: i64, rtt: i64) -> i64 {
    static_assert_uimm_bits!(IU3, 3);
    hexagon_S2_vspliceib(rss, rtt, IU3 as i32)
}

/// `Rdd32=vspliceb(Rss32,Rtt32,Pu4)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vspliceb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vspliceb_PPp(rss: i64, rtt: i64, pu: i32) -> i64 {
    hexagon_S2_vsplicerb(rss, rtt, pu)
}

/// `Rdd32=vsxtbh(Rs32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vsxtbh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vsxtbh_R(rs: i32) -> i64 {
    hexagon_S2_vsxtbh(rs)
}

/// `Rdd32=vsxthw(Rs32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vsxthw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vsxthw_R(rs: i32) -> i64 {
    hexagon_S2_vsxthw(rs)
}

/// `Rd32=vtrunehb(Rss32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vtrunehb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_vtrunehb_P(rss: i64) -> i32 {
    hexagon_S2_vtrunehb(rss)
}

/// `Rdd32=vtrunewh(Rss32,Rtt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vtrunewh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vtrunewh_PP(rss: i64, rtt: i64) -> i64 {
    hexagon_S2_vtrunewh(rss, rtt)
}

/// `Rd32=vtrunohb(Rss32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vtrunohb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_vtrunohb_P(rss: i64) -> i32 {
    hexagon_S2_vtrunohb(rss)
}

/// `Rdd32=vtrunowh(Rss32,Rtt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vtrunowh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vtrunowh_PP(rss: i64, rtt: i64) -> i64 {
    hexagon_S2_vtrunowh(rss, rtt)
}

/// `Rdd32=vzxtbh(Rs32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vzxtbh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vzxtbh_R(rs: i32) -> i64 {
    hexagon_S2_vzxtbh(rs)
}

/// `Rdd32=vzxthw(Rs32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vzxthw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vzxthw_R(rs: i32) -> i64 {
    hexagon_S2_vzxthw(rs)
}

/// `Rd32=add(Rs32,add(Ru32,#s6))`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(add, IS6 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_add_add_RRI<const IS6: i32>(rs: i32, ru: i32) -> i32 {
    static_assert_simm_bits!(IS6, 6);
    hexagon_S4_addaddi(rs, ru, IS6)
}

/// `Rx32=add(#u8,asl(Rx32,#U5))`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(0, 2)]
#[cfg_attr(test, assert_instr(add, IU8 = 0, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_add_asl_IRI<const IU8: u32, const IU5: u32>(rx: i32) -> i32 {
    static_assert_uimm_bits!(IU8, 8);
    static_assert_uimm_bits!(IU5, 5);
    hexagon_S4_addi_asl_ri(IU8 as i32, rx, IU5 as i32)
}

/// `Rx32=add(#u8,lsr(Rx32,#U5))`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(0, 2)]
#[cfg_attr(test, assert_instr(add, IU8 = 0, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_add_lsr_IRI<const IU8: u32, const IU5: u32>(rx: i32) -> i32 {
    static_assert_uimm_bits!(IU8, 8);
    static_assert_uimm_bits!(IU5, 5);
    hexagon_S4_addi_lsr_ri(IU8 as i32, rx, IU5 as i32)
}

/// `Rx32=and(#u8,asl(Rx32,#U5))`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(0, 2)]
#[cfg_attr(test, assert_instr(and, IU8 = 0, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_and_asl_IRI<const IU8: u32, const IU5: u32>(rx: i32) -> i32 {
    static_assert_uimm_bits!(IU8, 8);
    static_assert_uimm_bits!(IU5, 5);
    hexagon_S4_andi_asl_ri(IU8 as i32, rx, IU5 as i32)
}

/// `Rx32=and(#u8,lsr(Rx32,#U5))`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(0, 2)]
#[cfg_attr(test, assert_instr(and, IU8 = 0, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_and_lsr_IRI<const IU8: u32, const IU5: u32>(rx: i32) -> i32 {
    static_assert_uimm_bits!(IU8, 8);
    static_assert_uimm_bits!(IU5, 5);
    hexagon_S4_andi_lsr_ri(IU8 as i32, rx, IU5 as i32)
}

/// `Rd32=add(clb(Rs32),#s6)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(add, IS6 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_add_clb_RI<const IS6: i32>(rs: i32) -> i32 {
    static_assert_simm_bits!(IS6, 6);
    hexagon_S4_clbaddi(rs, IS6)
}

/// `Rd32=add(clb(Rss32),#s6)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(add, IS6 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_add_clb_PI<const IS6: i32>(rss: i64) -> i32 {
    static_assert_simm_bits!(IS6, 6);
    hexagon_S4_clbpaddi(rss, IS6)
}

/// `Rd32=normamt(Rss32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(normamt))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_normamt_P(rss: i64) -> i32 {
    hexagon_S4_clbpnorm(rss)
}

/// `Rd32=extract(Rs32,#u5,#U5)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(1, 2)]
#[cfg_attr(test, assert_instr(extract, IU5 = 0, IU5_2 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_extract_RII<const IU5: u32, const IU5_2: u32>(rs: i32) -> i32 {
    static_assert_uimm_bits!(IU5, 5);
    static_assert_uimm_bits!(IU5_2, 5);
    hexagon_S4_extract(rs, IU5 as i32, IU5_2 as i32)
}

/// `Rd32=extract(Rs32,Rtt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(extract))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_extract_RP(rs: i32, rtt: i64) -> i32 {
    hexagon_S4_extract_rp(rs, rtt)
}

/// `Rdd32=extract(Rss32,#u6,#U6)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(1, 2)]
#[cfg_attr(test, assert_instr(extract, IU6 = 0, IU6_2 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_extract_PII<const IU6: u32, const IU6_2: u32>(rss: i64) -> i64 {
    static_assert_uimm_bits!(IU6, 6);
    static_assert_uimm_bits!(IU6_2, 6);
    hexagon_S4_extractp(rss, IU6 as i32, IU6_2 as i32)
}

/// `Rdd32=extract(Rss32,Rtt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(extract))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_extract_PP(rss: i64, rtt: i64) -> i64 {
    hexagon_S4_extractp_rp(rss, rtt)
}

/// `Rd32=lsl(#s6,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(0)]
#[cfg_attr(test, assert_instr(lsl, IS6 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_lsl_IR<const IS6: i32>(rt: i32) -> i32 {
    static_assert_simm_bits!(IS6, 6);
    hexagon_S4_lsli(IS6, rt)
}

/// `Pd4=!tstbit(Rs32,#u5)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_not_tstbit_RI<const IU5: u32>(rs: i32) -> i32 {
    static_assert_uimm_bits!(IU5, 5);
    hexagon_S4_ntstbit_i(rs, IU5 as i32)
}

/// `Pd4=!tstbit(Rs32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_not_tstbit_RR(rs: i32, rt: i32) -> i32 {
    hexagon_S4_ntstbit_r(rs, rt)
}

/// `Rx32|=and(Rs32,#s10)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(and, IS10 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_andor_RI<const IS10: i32>(rx: i32, rs: i32) -> i32 {
    static_assert_simm_bits!(IS10, 10);
    hexagon_S4_or_andi(rx, rs, IS10)
}

/// `Rx32=or(Ru32,and(Rx32,#s10))`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(or, IS10 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_or_and_RRI<const IS10: i32>(ru: i32, rx: i32) -> i32 {
    static_assert_simm_bits!(IS10, 10);
    hexagon_S4_or_andix(ru, rx, IS10)
}

/// `Rx32|=or(Rs32,#s10)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(or, IS10 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_oror_RI<const IS10: i32>(rx: i32, rs: i32) -> i32 {
    static_assert_simm_bits!(IS10, 10);
    hexagon_S4_or_ori(rx, rs, IS10)
}

/// `Rx32=or(#u8,asl(Rx32,#U5))`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(0, 2)]
#[cfg_attr(test, assert_instr(or, IU8 = 0, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_or_asl_IRI<const IU8: u32, const IU5: u32>(rx: i32) -> i32 {
    static_assert_uimm_bits!(IU8, 8);
    static_assert_uimm_bits!(IU5, 5);
    hexagon_S4_ori_asl_ri(IU8 as i32, rx, IU5 as i32)
}

/// `Rx32=or(#u8,lsr(Rx32,#U5))`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(0, 2)]
#[cfg_attr(test, assert_instr(or, IU8 = 0, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_or_lsr_IRI<const IU8: u32, const IU5: u32>(rx: i32) -> i32 {
    static_assert_uimm_bits!(IU8, 8);
    static_assert_uimm_bits!(IU5, 5);
    hexagon_S4_ori_lsr_ri(IU8 as i32, rx, IU5 as i32)
}

/// `Rd32=parity(Rs32,Rt32)`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(parity))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_parity_RR(rs: i32, rt: i32) -> i32 {
    hexagon_S4_parity(rs, rt)
}

/// `Rd32=add(Rs32,sub(#s6,Ru32))`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(add, IS6 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_add_sub_RIR<const IS6: i32>(rs: i32, ru: i32) -> i32 {
    static_assert_simm_bits!(IS6, 6);
    hexagon_S4_subaddi(rs, IS6, ru)
}

/// `Rx32=sub(#u8,asl(Rx32,#U5))`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(0, 2)]
#[cfg_attr(test, assert_instr(sub, IU8 = 0, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_sub_asl_IRI<const IU8: u32, const IU5: u32>(rx: i32) -> i32 {
    static_assert_uimm_bits!(IU8, 8);
    static_assert_uimm_bits!(IU5, 5);
    hexagon_S4_subi_asl_ri(IU8 as i32, rx, IU5 as i32)
}

/// `Rx32=sub(#u8,lsr(Rx32,#U5))`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(0, 2)]
#[cfg_attr(test, assert_instr(sub, IU8 = 0, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_sub_lsr_IRI<const IU8: u32, const IU5: u32>(rx: i32) -> i32 {
    static_assert_uimm_bits!(IU8, 8);
    static_assert_uimm_bits!(IU5, 5);
    hexagon_S4_subi_lsr_ri(IU8 as i32, rx, IU5 as i32)
}

/// `Rdd32=vrcrotate(Rss32,Rt32,#u2)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(vrcrotate, IU2 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vrcrotate_PRI<const IU2: u32>(rss: i64, rt: i32) -> i64 {
    static_assert_uimm_bits!(IU2, 2);
    hexagon_S4_vrcrotate(rss, rt, IU2 as i32)
}

/// `Rxx32+=vrcrotate(Rss32,Rt32,#u2)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(3)]
#[cfg_attr(test, assert_instr(vrcrotate, IU2 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vrcrotateacc_PRI<const IU2: u32>(rxx: i64, rss: i64, rt: i32) -> i64 {
    static_assert_uimm_bits!(IU2, 2);
    hexagon_S4_vrcrotate_acc(rxx, rss, rt, IU2 as i32)
}

/// `Rdd32=vxaddsubh(Rss32,Rtt32):sat`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vxaddsubh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vxaddsubh_PP_sat(rss: i64, rtt: i64) -> i64 {
    hexagon_S4_vxaddsubh(rss, rtt)
}

/// `Rdd32=vxaddsubh(Rss32,Rtt32):rnd:>>1:sat`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vxaddsubh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vxaddsubh_PP_rnd_rs1_sat(rss: i64, rtt: i64) -> i64 {
    hexagon_S4_vxaddsubhr(rss, rtt)
}

/// `Rdd32=vxaddsubw(Rss32,Rtt32):sat`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vxaddsubw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vxaddsubw_PP_sat(rss: i64, rtt: i64) -> i64 {
    hexagon_S4_vxaddsubw(rss, rtt)
}

/// `Rdd32=vxsubaddh(Rss32,Rtt32):sat`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vxsubaddh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vxsubaddh_PP_sat(rss: i64, rtt: i64) -> i64 {
    hexagon_S4_vxsubaddh(rss, rtt)
}

/// `Rdd32=vxsubaddh(Rss32,Rtt32):rnd:>>1:sat`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vxsubaddh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vxsubaddh_PP_rnd_rs1_sat(rss: i64, rtt: i64) -> i64 {
    hexagon_S4_vxsubaddhr(rss, rtt)
}

/// `Rdd32=vxsubaddw(Rss32,Rtt32):sat`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(vxsubaddw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vxsubaddw_PP_sat(rss: i64, rtt: i64) -> i64 {
    hexagon_S4_vxsubaddw(rss, rtt)
}

/// `Rd32=vasrhub(Rss32,#u4):rnd:sat`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT0123
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(vasrhub, IU4 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_vasrhub_PI_rnd_sat<const IU4: u32>(rss: i64) -> i32 {
    static_assert_uimm_bits!(IU4, 4);
    hexagon_S5_asrhub_rnd_sat_goodsyntax(rss, IU4 as i32)
}

/// `Rd32=vasrhub(Rss32,#u4):sat`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(vasrhub, IU4 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_vasrhub_PI_sat<const IU4: u32>(rss: i64) -> i32 {
    static_assert_uimm_bits!(IU4, 4);
    hexagon_S5_asrhub_sat(rss, IU4 as i32)
}

/// `Rd32=popcount(Rss32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
#[inline(always)]
#[cfg_attr(test, assert_instr(popcount))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_popcount_P(rss: i64) -> i32 {
    hexagon_S5_popcountp(rss)
}

/// `Rdd32=vasrh(Rss32,#u4):rnd`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT0123
#[inline(always)]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(vasrh, IU4 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vasrh_PI_rnd<const IU4: u32>(rss: i64) -> i64 {
    static_assert_uimm_bits!(IU4, 4);
    hexagon_S5_vasrhrnd_goodsyntax(rss, IU4 as i32)
}

/// `dccleana(Rs32)`
///
/// Instruction Type: ST
/// Execution Slots: SLOT0
#[inline(always)]
#[cfg_attr(test, assert_instr(dccleana))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_dccleana_A(rs: i32) {
    hexagon_Y2_dccleana(rs)
}

/// `dccleaninva(Rs32)`
///
/// Instruction Type: ST
/// Execution Slots: SLOT0
#[inline(always)]
#[cfg_attr(test, assert_instr(dccleaninva))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_dccleaninva_A(rs: i32) {
    hexagon_Y2_dccleaninva(rs)
}

/// `dcfetch(Rs32)`
///
/// Instruction Type: MAPPING
/// Execution Slots: SLOT0123
#[inline(always)]
#[cfg_attr(test, assert_instr(dcfetch))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_dcfetch_A(rs: i32) {
    hexagon_Y2_dcfetch(rs)
}

/// `dcinva(Rs32)`
///
/// Instruction Type: ST
/// Execution Slots: SLOT0
#[inline(always)]
#[cfg_attr(test, assert_instr(dcinva))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_dcinva_A(rs: i32) {
    hexagon_Y2_dcinva(rs)
}

/// `dczeroa(Rs32)`
///
/// Instruction Type: ST
/// Execution Slots: SLOT0
#[inline(always)]
#[cfg_attr(test, assert_instr(dczeroa))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_dczeroa_A(rs: i32) {
    hexagon_Y2_dczeroa(rs)
}

/// `l2fetch(Rs32,Rt32)`
///
/// Instruction Type: ST
/// Execution Slots: SLOT0
#[inline(always)]
#[cfg_attr(test, assert_instr(l2fetch))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_l2fetch_AR(rs: i32, rt: i32) {
    hexagon_Y4_l2fetch(rs, rt)
}

/// `l2fetch(Rs32,Rtt32)`
///
/// Instruction Type: ST
/// Execution Slots: SLOT0
#[inline(always)]
#[cfg_attr(test, assert_instr(l2fetch))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_l2fetch_AP(rs: i32, rtt: i64) {
    hexagon_Y5_l2fetch(rs, rtt)
}

/// `Rdd32=rol(Rss32,#u6)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
/// Requires: V60
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v60"))]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(rol, IU6 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_rol_PI<const IU6: u32>(rss: i64) -> i64 {
    static_assert_uimm_bits!(IU6, 6);
    hexagon_S6_rol_i_p(rss, IU6 as i32)
}

/// `Rxx32+=rol(Rss32,#u6)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
/// Requires: V60
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v60"))]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(rol, IU6 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_rolacc_PI<const IU6: u32>(rxx: i64, rss: i64) -> i64 {
    static_assert_uimm_bits!(IU6, 6);
    hexagon_S6_rol_i_p_acc(rxx, rss, IU6 as i32)
}

/// `Rxx32&=rol(Rss32,#u6)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
/// Requires: V60
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v60"))]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(rol, IU6 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_roland_PI<const IU6: u32>(rxx: i64, rss: i64) -> i64 {
    static_assert_uimm_bits!(IU6, 6);
    hexagon_S6_rol_i_p_and(rxx, rss, IU6 as i32)
}

/// `Rxx32-=rol(Rss32,#u6)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
/// Requires: V60
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v60"))]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(rol, IU6 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_rolnac_PI<const IU6: u32>(rxx: i64, rss: i64) -> i64 {
    static_assert_uimm_bits!(IU6, 6);
    hexagon_S6_rol_i_p_nac(rxx, rss, IU6 as i32)
}

/// `Rxx32|=rol(Rss32,#u6)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
/// Requires: V60
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v60"))]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(rol, IU6 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_rolor_PI<const IU6: u32>(rxx: i64, rss: i64) -> i64 {
    static_assert_uimm_bits!(IU6, 6);
    hexagon_S6_rol_i_p_or(rxx, rss, IU6 as i32)
}

/// `Rxx32^=rol(Rss32,#u6)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
/// Requires: V60
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v60"))]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(rol, IU6 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_rolxacc_PI<const IU6: u32>(rxx: i64, rss: i64) -> i64 {
    static_assert_uimm_bits!(IU6, 6);
    hexagon_S6_rol_i_p_xacc(rxx, rss, IU6 as i32)
}

/// `Rd32=rol(Rs32,#u5)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
/// Requires: V60
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v60"))]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(rol, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_rol_RI<const IU5: u32>(rs: i32) -> i32 {
    static_assert_uimm_bits!(IU5, 5);
    hexagon_S6_rol_i_r(rs, IU5 as i32)
}

/// `Rx32+=rol(Rs32,#u5)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
/// Requires: V60
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v60"))]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(rol, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_rolacc_RI<const IU5: u32>(rx: i32, rs: i32) -> i32 {
    static_assert_uimm_bits!(IU5, 5);
    hexagon_S6_rol_i_r_acc(rx, rs, IU5 as i32)
}

/// `Rx32&=rol(Rs32,#u5)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
/// Requires: V60
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v60"))]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(rol, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_roland_RI<const IU5: u32>(rx: i32, rs: i32) -> i32 {
    static_assert_uimm_bits!(IU5, 5);
    hexagon_S6_rol_i_r_and(rx, rs, IU5 as i32)
}

/// `Rx32-=rol(Rs32,#u5)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
/// Requires: V60
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v60"))]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(rol, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_rolnac_RI<const IU5: u32>(rx: i32, rs: i32) -> i32 {
    static_assert_uimm_bits!(IU5, 5);
    hexagon_S6_rol_i_r_nac(rx, rs, IU5 as i32)
}

/// `Rx32|=rol(Rs32,#u5)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
/// Requires: V60
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v60"))]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(rol, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_rolor_RI<const IU5: u32>(rx: i32, rs: i32) -> i32 {
    static_assert_uimm_bits!(IU5, 5);
    hexagon_S6_rol_i_r_or(rx, rs, IU5 as i32)
}

/// `Rx32^=rol(Rs32,#u5)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
/// Requires: V60
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v60"))]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(test, assert_instr(rol, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_rolxacc_RI<const IU5: u32>(rx: i32, rs: i32) -> i32 {
    static_assert_uimm_bits!(IU5, 5);
    hexagon_S6_rol_i_r_xacc(rx, rs, IU5 as i32)
}

/// `Rdd32=vabsdiffb(Rtt32,Rss32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
/// Requires: V62
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v62"))]
#[cfg_attr(test, assert_instr(vabsdiffb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vabsdiffb_PP(rtt: i64, rss: i64) -> i64 {
    hexagon_M6_vabsdiffb(rtt, rss)
}

/// `Rdd32=vabsdiffub(Rtt32,Rss32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
/// Requires: V62
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v62"))]
#[cfg_attr(test, assert_instr(vabsdiffub))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vabsdiffub_PP(rtt: i64, rss: i64) -> i64 {
    hexagon_M6_vabsdiffub(rtt, rss)
}

/// `Rdd32=vsplatb(Rs32)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
/// Requires: V62
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v62"))]
#[cfg_attr(test, assert_instr(vsplatb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vsplatb_R(rs: i32) -> i64 {
    hexagon_S6_vsplatrbp(rs)
}

/// `Rdd32=vtrunehb(Rss32,Rtt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
/// Requires: V62
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v62"))]
#[cfg_attr(test, assert_instr(vtrunehb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vtrunehb_PP(rss: i64, rtt: i64) -> i64 {
    hexagon_S6_vtrunehb_ppp(rss, rtt)
}

/// `Rdd32=vtrunohb(Rss32,Rtt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
/// Requires: V62
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v62"))]
#[cfg_attr(test, assert_instr(vtrunohb))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vtrunohb_PP(rss: i64, rtt: i64) -> i64 {
    hexagon_S6_vtrunohb_ppp(rss, rtt)
}

/// `Pd4=!any8(vcmpb.eq(Rss32,Rtt32))`
///
/// Instruction Type: ALU64
/// Execution Slots: SLOT23
/// Requires: V65
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v65"))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_p_not_any8_vcmpb_eq_PP(rss: i64, rtt: i64) -> i32 {
    hexagon_A6_vcmpbeq_notany(rss, rtt)
}

/// `Rdd32=dfadd(Rss32,Rtt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
/// Requires: V66
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v66"))]
#[cfg_attr(test, assert_instr(dfadd))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_dfadd_PP(rss: f64, rtt: f64) -> f64 {
    hexagon_F2_dfadd(rss, rtt)
}

/// `Rdd32=dfsub(Rss32,Rtt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
/// Requires: V66
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v66"))]
#[cfg_attr(test, assert_instr(dfsub))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_dfsub_PP(rss: f64, rtt: f64) -> f64 {
    hexagon_F2_dfsub(rss, rtt)
}

/// `Rx32-=mpyi(Rs32,Rt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
/// Requires: V66
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v66"))]
#[cfg_attr(test, assert_instr(mpyi))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mpyinac_RR(rx: i32, rs: i32, rt: i32) -> i32 {
    hexagon_M2_mnaci(rx, rs, rt)
}

/// `Rd32=mask(#u5,#U5)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
/// Requires: V66
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v66"))]
#[rustc_legacy_const_generics(0, 1)]
#[cfg_attr(test, assert_instr(mask, IU5 = 0, IU5_2 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_mask_II<const IU5: u32, const IU5_2: u32>() -> i32 {
    static_assert_uimm_bits!(IU5, 5);
    static_assert_uimm_bits!(IU5_2, 5);
    hexagon_S2_mask(IU5 as i32, IU5_2 as i32)
}

/// `Rd32=clip(Rs32,#u5)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
/// Requires: V67, Audio
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v67,audio"))]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(clip, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_clip_RI<const IU5: u32>(rs: i32) -> i32 {
    static_assert_uimm_bits!(IU5, 5);
    hexagon_A7_clip(rs, IU5 as i32)
}

/// `Rdd32=cround(Rss32,#u6)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
/// Requires: V67, Audio
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v67,audio"))]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(cround, IU6 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_cround_PI<const IU6: u32>(rss: i64) -> i64 {
    static_assert_uimm_bits!(IU6, 6);
    hexagon_A7_croundd_ri(rss, IU6 as i32)
}

/// `Rdd32=cround(Rss32,Rt32)`
///
/// Instruction Type: S_3op
/// Execution Slots: SLOT23
/// Requires: V67, Audio
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v67,audio"))]
#[cfg_attr(test, assert_instr(cround))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_cround_PR(rss: i64, rt: i32) -> i64 {
    hexagon_A7_croundd_rr(rss, rt)
}

/// `Rdd32=vclip(Rss32,#u5)`
///
/// Instruction Type: S_2op
/// Execution Slots: SLOT23
/// Requires: V67, Audio
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v67,audio"))]
#[rustc_legacy_const_generics(1)]
#[cfg_attr(test, assert_instr(vclip, IU5 = 0))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vclip_PI<const IU5: u32>(rss: i64) -> i64 {
    static_assert_uimm_bits!(IU5, 5);
    hexagon_A7_vclip(rss, IU5 as i32)
}

/// `Rdd32=dfmax(Rss32,Rtt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
/// Requires: V67
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v67"))]
#[cfg_attr(test, assert_instr(dfmax))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_dfmax_PP(rss: f64, rtt: f64) -> f64 {
    hexagon_F2_dfmax(rss, rtt)
}

/// `Rdd32=dfmin(Rss32,Rtt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
/// Requires: V67
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v67"))]
#[cfg_attr(test, assert_instr(dfmin))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_dfmin_PP(rss: f64, rtt: f64) -> f64 {
    hexagon_F2_dfmin(rss, rtt)
}

/// `Rdd32=dfmpyfix(Rss32,Rtt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
/// Requires: V67
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v67"))]
#[cfg_attr(test, assert_instr(dfmpyfix))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_dfmpyfix_PP(rss: f64, rtt: f64) -> f64 {
    hexagon_F2_dfmpyfix(rss, rtt)
}

/// `Rxx32+=dfmpyhh(Rss32,Rtt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
/// Requires: V67
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v67"))]
#[cfg_attr(test, assert_instr(dfmpyhh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_dfmpyhhacc_PP(rxx: f64, rss: f64, rtt: f64) -> f64 {
    hexagon_F2_dfmpyhh(rxx, rss, rtt)
}

/// `Rxx32+=dfmpylh(Rss32,Rtt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
/// Requires: V67
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v67"))]
#[cfg_attr(test, assert_instr(dfmpylh))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_dfmpylhacc_PP(rxx: f64, rss: f64, rtt: f64) -> f64 {
    hexagon_F2_dfmpylh(rxx, rss, rtt)
}

/// `Rdd32=dfmpyll(Rss32,Rtt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT23
/// Requires: V67
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v67"))]
#[cfg_attr(test, assert_instr(dfmpyll))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_dfmpyll_PP(rss: f64, rtt: f64) -> f64 {
    hexagon_F2_dfmpyll(rss, rtt)
}

/// `Rdd32=cmpyiw(Rss32,Rtt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT3
/// Requires: V67, Audio
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v67,audio"))]
#[cfg_attr(test, assert_instr(cmpyiw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_cmpyiw_PP(rss: i64, rtt: i64) -> i64 {
    hexagon_M7_dcmpyiw(rss, rtt)
}

/// `Rxx32+=cmpyiw(Rss32,Rtt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT3
/// Requires: V67, Audio
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v67,audio"))]
#[cfg_attr(test, assert_instr(cmpyiw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_cmpyiwacc_PP(rxx: i64, rss: i64, rtt: i64) -> i64 {
    hexagon_M7_dcmpyiw_acc(rxx, rss, rtt)
}

/// `Rdd32=cmpyiw(Rss32,Rtt32*)`
///
/// Instruction Type: M
/// Execution Slots: SLOT3
/// Requires: V67, Audio
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v67,audio"))]
#[cfg_attr(test, assert_instr(cmpyiw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_cmpyiw_PP_conj(rss: i64, rtt: i64) -> i64 {
    hexagon_M7_dcmpyiwc(rss, rtt)
}

/// `Rxx32+=cmpyiw(Rss32,Rtt32*)`
///
/// Instruction Type: M
/// Execution Slots: SLOT3
/// Requires: V67, Audio
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v67,audio"))]
#[cfg_attr(test, assert_instr(cmpyiw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_cmpyiwacc_PP_conj(rxx: i64, rss: i64, rtt: i64) -> i64 {
    hexagon_M7_dcmpyiwc_acc(rxx, rss, rtt)
}

/// `Rdd32=cmpyrw(Rss32,Rtt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT3
/// Requires: V67, Audio
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v67,audio"))]
#[cfg_attr(test, assert_instr(cmpyrw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_cmpyrw_PP(rss: i64, rtt: i64) -> i64 {
    hexagon_M7_dcmpyrw(rss, rtt)
}

/// `Rxx32+=cmpyrw(Rss32,Rtt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT3
/// Requires: V67, Audio
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v67,audio"))]
#[cfg_attr(test, assert_instr(cmpyrw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_cmpyrwacc_PP(rxx: i64, rss: i64, rtt: i64) -> i64 {
    hexagon_M7_dcmpyrw_acc(rxx, rss, rtt)
}

/// `Rdd32=cmpyrw(Rss32,Rtt32*)`
///
/// Instruction Type: M
/// Execution Slots: SLOT3
/// Requires: V67, Audio
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v67,audio"))]
#[cfg_attr(test, assert_instr(cmpyrw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_cmpyrw_PP_conj(rss: i64, rtt: i64) -> i64 {
    hexagon_M7_dcmpyrwc(rss, rtt)
}

/// `Rxx32+=cmpyrw(Rss32,Rtt32*)`
///
/// Instruction Type: M
/// Execution Slots: SLOT3
/// Requires: V67, Audio
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v67,audio"))]
#[cfg_attr(test, assert_instr(cmpyrw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_cmpyrwacc_PP_conj(rxx: i64, rss: i64, rtt: i64) -> i64 {
    hexagon_M7_dcmpyrwc_acc(rxx, rss, rtt)
}

/// `Rdd32=vdmpyw(Rss32,Rtt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT3
/// Requires: V67, Audio
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v67,audio"))]
#[cfg_attr(test, assert_instr(vdmpyw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vdmpyw_PP(rss: i64, rtt: i64) -> i64 {
    hexagon_M7_vdmpy(rss, rtt)
}

/// `Rxx32+=vdmpyw(Rss32,Rtt32)`
///
/// Instruction Type: M
/// Execution Slots: SLOT3
/// Requires: V67, Audio
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v67,audio"))]
#[cfg_attr(test, assert_instr(vdmpyw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_P_vdmpywacc_PP(rxx: i64, rss: i64, rtt: i64) -> i64 {
    hexagon_M7_vdmpy_acc(rxx, rss, rtt)
}

/// `Rd32=cmpyiw(Rss32,Rtt32):<<1:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT3
/// Requires: V67, Audio
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v67,audio"))]
#[cfg_attr(test, assert_instr(cmpyiw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_cmpyiw_PP_s1_sat(rss: i64, rtt: i64) -> i32 {
    hexagon_M7_wcmpyiw(rss, rtt)
}

/// `Rd32=cmpyiw(Rss32,Rtt32):<<1:rnd:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT3
/// Requires: V67, Audio
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v67,audio"))]
#[cfg_attr(test, assert_instr(cmpyiw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_cmpyiw_PP_s1_rnd_sat(rss: i64, rtt: i64) -> i32 {
    hexagon_M7_wcmpyiw_rnd(rss, rtt)
}

/// `Rd32=cmpyiw(Rss32,Rtt32*):<<1:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT3
/// Requires: V67, Audio
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v67,audio"))]
#[cfg_attr(test, assert_instr(cmpyiw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_cmpyiw_PP_conj_s1_sat(rss: i64, rtt: i64) -> i32 {
    hexagon_M7_wcmpyiwc(rss, rtt)
}

/// `Rd32=cmpyiw(Rss32,Rtt32*):<<1:rnd:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT3
/// Requires: V67, Audio
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v67,audio"))]
#[cfg_attr(test, assert_instr(cmpyiw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_cmpyiw_PP_conj_s1_rnd_sat(rss: i64, rtt: i64) -> i32 {
    hexagon_M7_wcmpyiwc_rnd(rss, rtt)
}

/// `Rd32=cmpyrw(Rss32,Rtt32):<<1:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT3
/// Requires: V67, Audio
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v67,audio"))]
#[cfg_attr(test, assert_instr(cmpyrw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_cmpyrw_PP_s1_sat(rss: i64, rtt: i64) -> i32 {
    hexagon_M7_wcmpyrw(rss, rtt)
}

/// `Rd32=cmpyrw(Rss32,Rtt32):<<1:rnd:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT3
/// Requires: V67, Audio
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v67,audio"))]
#[cfg_attr(test, assert_instr(cmpyrw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_cmpyrw_PP_s1_rnd_sat(rss: i64, rtt: i64) -> i32 {
    hexagon_M7_wcmpyrw_rnd(rss, rtt)
}

/// `Rd32=cmpyrw(Rss32,Rtt32*):<<1:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT3
/// Requires: V67, Audio
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v67,audio"))]
#[cfg_attr(test, assert_instr(cmpyrw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_cmpyrw_PP_conj_s1_sat(rss: i64, rtt: i64) -> i32 {
    hexagon_M7_wcmpyrwc(rss, rtt)
}

/// `Rd32=cmpyrw(Rss32,Rtt32*):<<1:rnd:sat`
///
/// Instruction Type: M
/// Execution Slots: SLOT3
/// Requires: V67, Audio
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v67,audio"))]
#[cfg_attr(test, assert_instr(cmpyrw))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_cmpyrw_PP_conj_s1_rnd_sat(rss: i64, rtt: i64) -> i32 {
    hexagon_M7_wcmpyrwc_rnd(rss, rtt)
}

/// `dmlink(Rs32,Rt32)`
///
/// Instruction Type: ST
/// Execution Slots: SLOT0
/// Requires: V68
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v68"))]
#[cfg_attr(test, assert_instr(dmlink))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_dmlink_AA(rs: i32, rt: i32) {
    hexagon_Y6_dmlink(rs, rt)
}

/// `Rd32=dmpause`
///
/// Instruction Type: ST
/// Execution Slots: SLOT0
/// Requires: V68
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v68"))]
#[cfg_attr(test, assert_instr(dmpause))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_dmpause() -> i32 {
    hexagon_Y6_dmpause()
}

/// `Rd32=dmpoll`
///
/// Instruction Type: ST
/// Execution Slots: SLOT0
/// Requires: V68
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v68"))]
#[cfg_attr(test, assert_instr(dmpoll))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_dmpoll() -> i32 {
    hexagon_Y6_dmpoll()
}

/// `dmresume(Rs32)`
///
/// Instruction Type: ST
/// Execution Slots: SLOT0
/// Requires: V68
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v68"))]
#[cfg_attr(test, assert_instr(dmresume))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_dmresume_A(rs: i32) {
    hexagon_Y6_dmresume(rs)
}

/// `dmstart(Rs32)`
///
/// Instruction Type: ST
/// Execution Slots: SLOT0
/// Requires: V68
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v68"))]
#[cfg_attr(test, assert_instr(dmstart))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_dmstart_A(rs: i32) {
    hexagon_Y6_dmstart(rs)
}

/// `Rd32=dmwait`
///
/// Instruction Type: ST
/// Execution Slots: SLOT0
/// Requires: V68
#[inline(always)]
#[cfg_attr(target_arch = "hexagon", target_feature(enable = "v68"))]
#[cfg_attr(test, assert_instr(dmwait))]
#[unstable(feature = "stdarch_hexagon", issue = "151523")]
pub unsafe fn Q6_R_dmwait() -> i32 {
    hexagon_Y6_dmwait()
}
