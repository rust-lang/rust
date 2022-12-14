// run-pass

#![feature(naked_functions)]
#![allow(dead_code)]

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
core::arch::global_asm!(
    r#"
    .global foo
    .global _foo
foo:
_foo:
    ret
"#
);

extern "C" {
    fn foo();
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
fn main() {
    unsafe {
        foo();
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
fn main() {}
