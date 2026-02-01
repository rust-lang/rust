// Codegen test of mandatory Cortex-R82 extensions

//@ add-minicore
//@ compile-flags: --target aarch64v8r-unknown-none -C target-cpu=cortex-r82
//@ needs-llvm-components: aarch64
//@ build-pass
//@ ignore-backends: gcc

#![deny(dead_code)]
#![feature(no_core)]
#![no_core]
#![no_main]
#![crate_type = "rlib"]

extern crate minicore;
use minicore::*;

/* # Mandatory extensions
 *
 * A `//` comment indicates that the extension has no associated assembly instruction and cannot
 * be codegen tested
 * A `/*  */` comment indicates that the extension is being tested in the ISA level codegen test
 * (`tests/ui/asm/aarch64v8r.rs`)
 *
 * ## References:
 *
 * - Arm Cortex-R82 Processor Technical Reference Manual Revision r3p1 (102670_0301_06_en Issue 6)
 *   section 3.2.1 has the list of mandatory extensions
 * - Arm Architecture Reference Manual for A-profile architecture (ARM DDI 0487) -- has the
 *   mapping from features to instructions
 * - Feature names in A-profile architecture (109697_0100_02_en Version 1.0) -- overview of what
 *   each extension mean
 * */
pub fn mandatory_extensions() {
    // FEAT_GICv3
    // FEAT_GICv3p1
    // FEAT_GICv3_TDIR
    feat_pmuv3();
    // FEAT_ETMv4
    // FEAT_ETMv4p1
    // FEAT_ETMv4p2
    // FEAT_ETMv4p3
    // FEAT_ETMv4p4
    // FEAT_ETMv4p5
    /* FEAT_RAS */
    // FEAT_PCSRv8
    feat_ssbs();
    feat_ssbs2();
    // FEAT_CSV2
    // FEAT_CSV2_1p1
    // FEAT_CSV3
    feat_sb();
    feat_specres();
    feat_dgh();
    // FEAT_nTLBPA
    /* FEAT_CRC32 */
    /* FEAT_LSE */
    feat_rdm();
    /* FEAT_HPDS */
    /* FEAT_PAN */
    // FEAT_HAFDBS
    // FEAT_PMUv3p1
    // FEAT_TTCNP
    // FEAT_XNX
    /* FEAT_UAO */
    feat_pan2();
    feat_dpb();
    /* FEAT_Debugv8p2 */
    /* FEAT_ASMv8p2 */
    // FEAT_IESB
    feat_fp16();
    // FEAT_PCSRv8p2
    feat_dotprod();
    feat_fhm();
    feat_dpb2();
    /* FEAT_PAuth */
    // FEAT_PACQARMA3
    // FEAT_PAuth2
    // FEAT_FPAC
    // FEAT_FPACCOMBINE
    // FEAT_CONSTPACFIELD
    feat_jscvt();
    /* FEAT_LRCPC */
    feat_fcma();
    // FEAT_DoPD
    // FEAT_SEL2
    /* FEAT_S2FWB */
    /* FEAT_DIT */
    /* FEAT_IDST */
    /* FEAT_FlagM */
    /* FEAT_LSE2 */
    /* FEAT_LRCPC2 */
    /* FEAT_TLBIOS */
    /* FEAT_TLBIRANGE */
    /* FEAT_TTL */
    // FEAT_BBM
    // FEAT_CNTSC
    feat_rasv1p1();
    // FEAT_Debugv8p4
    feat_pmuv3p4();
    feat_trf();
    // FEAT_TTST
    // FEAT_E0PD
}

fn feat_pmuv3() {
    unsafe { asm!("mrs x0, PMCCFILTR_EL0") }
}

fn feat_ssbs() {
    unsafe { asm!("msr SSBS, 1") }
}

fn feat_ssbs2() {
    unsafe { asm!("mrs x0, SSBS") }
}

fn feat_sb() {
    unsafe { asm!("sb") }
}

fn feat_specres() {
    unsafe { asm!("cfp rctx, x0") }
}

fn feat_dgh() {
    unsafe { asm!("dgh") }
}

fn feat_rdm() {
    unsafe { asm!("sqrdmlah v0.4h, v1.4h, v2.4h") }
}

fn feat_pan2() {
    unsafe { asm!("AT S1E1RP, x0") }
}

fn feat_dpb() {
    unsafe { asm!("DC CVAP, x0") }
}

fn feat_fp16() {
    unsafe { asm!("fmulx h0, h1, h2") }
}

fn feat_dotprod() {
    unsafe { asm!("sdot V0.4S, V1.16B, V2.16B") }
}

fn feat_fhm() {
    unsafe { asm!("fmlal v0.2s, v1.2h, v2.2h") }
}

fn feat_dpb2() {
    unsafe { asm!("DC CVADP, x0") }
}

fn feat_jscvt() {
    unsafe { asm!("fjcvtzs w0, d1") }
}

fn feat_fcma() {
    unsafe { asm!("fcadd v0.4h, v1.4h, v2.4h, #90") }
}

fn feat_rasv1p1() {
    unsafe { asm!("mrs x0, ERXMISC2_EL1") }
}

fn feat_pmuv3p4() {
    unsafe { asm!("mrs x0, PMMIR_EL1") }
}

fn feat_trf() {
    unsafe { asm!("tsb csync") }
}
