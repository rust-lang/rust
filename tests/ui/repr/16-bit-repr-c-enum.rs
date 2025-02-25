//@ add-core-stubs
//@ build-pass
//@ revisions: avr msp430
//
//@ [avr] needs-llvm-components: avr
//@ [avr] compile-flags: --target=avr-none -C target-cpu=atmega328p --crate-type=rlib
//@ [msp430] needs-llvm-components: msp430
//@ [msp430] compile-flags: --target=msp430-none-elf --crate-type=rlib
#![feature(no_core, lang_items, intrinsics, staged_api, rustc_attrs)]
#![no_core]
#![crate_type = "lib"]
#![stable(feature = "intrinsics_for_test", since = "3.3.3")]
#![allow(dead_code)]

extern crate minicore;
use minicore::*;

// Test that the repr(C) attribute doesn't break compilation
// Previous bad assumption was that 32-bit enum default width is fine on msp430, avr
// But the width of the C int on these platforms is 16 bits, and C enums <= C int range
// so we want no more than that, usually. This resulted in errors like
// "layout decided on a larger discriminant type (I32) than typeck (I16)"
#[repr(C)]
enum Foo {
    Bar,
}

#[stable(feature = "intrinsics_for_test", since = "3.3.3")]
#[rustc_const_stable(feature = "intrinsics_for_test", since = "3.3.3")]
#[rustc_intrinsic]
const fn size_of<T>() -> usize;

const EXPECTED: usize = 2;
const ACTUAL: usize = size_of::<Foo>();
// Validate that the size is indeed 16 bits, to match this C static_assert:
/**
```c
#include <assert.h>
enum foo {
    BAR
};
int main(void)
{
    /* passes on msp430-elf-gcc */
    static_assert(sizeof(enum foo) == 2);
}
```
*/
const _: [(); EXPECTED] = [(); ACTUAL];
