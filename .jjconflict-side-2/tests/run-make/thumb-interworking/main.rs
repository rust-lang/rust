#![feature(no_core)]
#![no_core]
#![no_main]

extern crate minicore;
use minicore::*;

#[unsafe(no_mangle)]
fn entry() {
    arm();
    thumb();
}

#[unsafe(no_mangle)]
pub fn arm() {
    // thumbv5te-LABEL: <arm>:
    // thumbv5te: blx {{0x[0-9a-f]+}} <main::arm_normalfn>
    // thumbv5te: blx {{0x[0-9a-f]+}} <arm_globalfn>
    // thumbv5te: blx {{0x[0-9a-f]+}} <main::arm_nakedfn>

    // thumbv4t-LABEL: <arm>:
    // thumbv4t: bl {{0x[0-9a-f]+}} <__Thumbv4ABSLongBXThunk__{{.*}}arm_normalfn>
    // thumbv4t: bl {{0x[0-9a-f]+}} <__Thumbv4ABSLongBXThunk_arm_globalfn>
    // thumbv4t: bl {{0x[0-9a-f]+}} <__Thumbv4ABSLongBXThunk__{{.*}}arm_nakedfn>

    // armv5te-LABEL: <arm>:
    // armv5te: bl {{0x[0-9a-f]+}} <main::arm_normalfn>
    // armv5te: bl {{0x[0-9a-f]+}} <arm_globalfn>
    // armv5te: bl {{0x[0-9a-f]+}} <main::arm_nakedfn>

    // armv4t-LABEL: <arm>:
    // armv4t: bl {{0x[0-9a-f]+}} <main::arm_normalfn>
    // armv4t: bl {{0x[0-9a-f]+}} <arm_globalfn>
    // armv4t: bl {{0x[0-9a-f]+}} <main::arm_nakedfn>
    arm_normalfn();
    arm_globalfn();
    arm_nakedfn();
}

#[unsafe(no_mangle)]
pub fn thumb() {
    // thumbv5te-LABEL: <thumb>:
    // thumbv5te: bl {{0x[0-9a-f]+}} <main::thumb_normalfn>
    // thumbv5te: bl {{0x[0-9a-f]+}} <thumb_globalfn>
    // thumbv5te: bl {{0x[0-9a-f]+}} <main::thumb_nakedfn>

    // thumbv4t-LABEL: <thumb>:
    // thumbv4t: bl {{0x[0-9a-f]+}} <main::thumb_normalfn>
    // thumbv4t: bl {{0x[0-9a-f]+}} <thumb_globalfn>
    // thumbv4t: bl {{0x[0-9a-f]+}} <main::thumb_nakedfn>

    // armv5te-LABEL: <thumb>:
    // armv5te: blx {{0x[0-9a-f]+}} <main::thumb_normalfn>
    // armv5te: blx {{0x[0-9a-f]+}} <thumb_globalfn>
    // armv5te: blx {{0x[0-9a-f]+}} <main::thumb_nakedfn>

    // armv4t-LABEL: <thumb>:
    // armv4t: bl {{0x[0-9a-f]+}} <__ARMv4ABSLongBXThunk__{{.*}}thumb_normalfn>
    // armv4t: bl {{0x[0-9a-f]+}} <__ARMv4ABSLongBXThunk_thumb_globalfn>
    // armv4t: bl {{0x[0-9a-f]+}} <__ARMv4ABSLongBXThunk__{{.*}}thumb_nakedfn>
    thumb_normalfn();
    thumb_globalfn();
    thumb_nakedfn();
}

#[instruction_set(arm::t32)]
extern "C" fn thumb_normalfn() {
    unsafe { asm!("nop") }
}

unsafe extern "C" {
    safe fn thumb_globalfn();
}

global_asm!(
    r#"
    .thumb
    .global thumb_globalfn
    .type thumb_globalfn, %function
    thumb_globalfn:
        nop
        bx lr
    .size thumb_globalfn, . - thumb_globalfn
"#
);

#[unsafe(naked)]
#[instruction_set(arm::t32)]
extern "C" fn thumb_nakedfn() {
    naked_asm!("nop", "bx lr",);
}

#[instruction_set(arm::a32)]
extern "C" fn arm_normalfn() {
    unsafe { asm!("nop") }
}

unsafe extern "C" {
    safe fn arm_globalfn();
}

global_asm!(
    r#"
    .arm
    .global arm_globalfn
    .type arm_globalfn, %function
    arm_globalfn:
        nop
        bx lr
    .size arm_globalfn, . - arm_globalfn
"#
);

#[unsafe(naked)]
#[instruction_set(arm::a32)]
extern "C" fn arm_nakedfn() {
    naked_asm!("nop", "bx lr",);
}
