//@ compile-flags: -O
//@ compile-flags: --target aarch64-unknown-linux-gnu
//@ needs-llvm-components: aarch64

#![feature(no_core, lang_items, intrinsics)]
#![no_core]
#![crate_type = "lib"]

#[lang = "sized"]
pub trait Sized {}
#[lang = "copy"]
pub trait Copy {}

impl Copy for u32 {}
impl<T> Copy for *mut T {}

extern "rust-intrinsic" {
    pub fn nontemporal_store<T>(ptr: *mut T, val: T);
}

#[no_mangle]
pub fn a(a: &mut u32, b: u32) {
    // CHECK-LABEL: define{{.*}}void @a
    // CHECK: store i32 %b, ptr %a, align 4, !nontemporal
    unsafe {
        nontemporal_store(a, b);
    }
}
