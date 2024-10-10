// We should fail to compute alignment for types aligned higher than usize::MAX.
// We can't handle alignments that require all 32 bits, so this only affects 16-bit.

//@ revisions: msp430 aarch32
//@ [msp430] build-fail
//@ [msp430] needs-llvm-components: msp430
//@ [msp430] compile-flags: --target=msp430-none-elf
//@ [msp430] error-pattern: values of the type `Hello16BitAlign` are too big for the target architecture

//@ [aarch32] build-pass
//@ [aarch32] needs-llvm-components: arm
//@ [aarch32] compile-flags: --target=thumbv7m-none-eabi

#![feature(no_core, lang_items, intrinsics, staged_api, rustc_attrs)]
#![no_core]
#![crate_type = "lib"]
#![stable(feature = "intrinsics_for_test", since = "3.3.3")]
#![allow(dead_code)]

extern "rust-intrinsic" {
    #[stable(feature = "intrinsics_for_test", since = "3.3.3")]
    #[rustc_const_stable(feature = "intrinsics_for_test", since = "3.3.3")]
    #[rustc_safe_intrinsic]
    fn min_align_of<T>() -> usize;
}

#[lang="sized"]
trait Sized {}
#[lang="copy"]
trait Copy {}

#[repr(align(65536))]
#[stable(feature = "intrinsics_for_test", since = "3.3.3")]
pub struct Hello16BitAlign;


#[stable(feature = "intrinsics_for_test", since = "3.3.3")]
pub fn bar() -> usize { min_align_of::<Hello16BitAlign>() }
