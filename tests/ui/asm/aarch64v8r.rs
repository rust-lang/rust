// Codegen test of mandatory Armv8-R AArch64 extensions

//@ add-minicore
//@ revisions: hf sf
//@ [hf] compile-flags: --target aarch64v8r-unknown-none
//@ [hf] needs-llvm-components: aarch64
//@ [sf] compile-flags: --target aarch64v8r-unknown-none-softfloat
//@ [sf] needs-llvm-components: aarch64
//@ build-pass
//@ ignore-backends: gcc

#![feature(no_core)]
#![no_core]
#![no_main]
#![crate_type = "rlib"]
#![deny(dead_code)] // ensures we call all private functions from the public one

extern crate minicore;
use minicore::*;

/* # Mandatory extensions
 *
 * A comment indicates that the extension has no associated assembly instruction and cannot be
 * codegen tested
 *
 * ## References:
 *
 * - Arm Architecture Reference Manual for R-profile AArch64 architecture (DDI 0628) -- has the
 *   list of mandatory extensions
 * - Arm Architecture Reference Manual for A-profile architecture (ARM DDI 0487) -- has the
 *   mapping from features to instructions
 * - Feature names in A-profile architecture (109697_0100_02_en Version 1.0) -- overview of
 *   what each extension mean
 * */
pub fn mandatory_extensions() {
    /* ## ARMv8.0 */
    feat_aa64();
    // FEAT_AA64EL0
    // FEAT_AA64EL1
    // FEAT_AA64EL2
    feat_crc32();
    // FEAT_EL0
    // FEAT_EL1
    // FEAT_EL2
    // FEAT_IVIPT

    /* ## ARMv8.1 */
    // FEAT_HPDS
    feat_lse();
    feat_pan();

    /* ## ARMv8.2 */
    feat_asmv8p2();
    feat_dpb();
    // FEAT_Debugv8p2
    // FEAT_PAN2
    feat_ras();
    // FEAT_TTCNP
    feat_uao();
    // FEAT_XNX

    /* ## ARMv8.3 */
    feat_lrcpc();
    feat_pauth();

    /* ## ARMv8.4 */
    feat_dit();
    // FEAT_Debugv8p4
    feat_flagm();
    // FEAT_IDST
    feat_lrcpc2();
    // FEAT_LSE2
    // FEAT_S2FWB
    feat_tlbios();
    feat_tlbirange();
    // FEAT_TTL
}

fn feat_aa64() {
    // CurrentEL register only present when FEAT_AA64 is implemented
    unsafe { asm!("mrs x0, CurrentEL") }
}

fn feat_crc32() {
    // instruction is present when FEAT_CRC32 is implemented
    unsafe { asm!("crc32b w0, w1, w2") }
}

fn feat_lse() {
    // instruction is present when FEAT_LSE is implemented
    unsafe { asm!("casp w0, w1, w2, w3, [x4]") }
}

fn feat_pan() {
    unsafe { asm!("mrs x0, PAN") }
}

fn feat_asmv8p2() {
    unsafe { asm!("BFC w0, #0, #1") }
}

fn feat_dpb() {
    unsafe { asm!("DC CVAP, x0") }
}

fn feat_ras() {
    unsafe { asm!("ESB") }
}

fn feat_uao() {
    unsafe { asm!("mrs x0, UAO") }
}

fn feat_lrcpc() {
    unsafe { asm!("ldaprb w0, [x1]") }
}

fn feat_pauth() {
    unsafe { asm!("xpacd x0") }
}

fn feat_dit() {
    unsafe { asm!("mrs x0, DIT") }
}

fn feat_flagm() {
    unsafe { asm!("cfinv") }
}

fn feat_lrcpc2() {
    unsafe { asm!("stlurb w0, [x1]") }
}

fn feat_tlbios() {
    unsafe { asm!("tlbi VMALLE1OS") }
}

fn feat_tlbirange() {
    unsafe { asm!("tlbi RVAE1IS, x0") }
}
