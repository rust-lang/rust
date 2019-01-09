#![feature(global_asm)]
#![feature(naked_functions)]
#![allow(dead_code)]

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
global_asm!(r#"
    .global foo
    .global _foo
foo:
_foo:
    ret
"#);

extern {
    fn foo();
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
fn main() { unsafe { foo(); } }

#[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
fn main() {}
