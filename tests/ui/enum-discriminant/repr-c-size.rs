//@ check-pass
//@ revisions: linux32 linux64 msvc32 msvc64
//@[linux32] compile-flags: --target i686-unknown-linux-gnu
//@[linux32] needs-llvm-components: x86
//@[linux64] compile-flags: --target x86_64-unknown-linux-gnu
//@[linux64] needs-llvm-components: x86
//@[msvc32] compile-flags: --target i686-pc-windows-msvc
//@[msvc32] needs-llvm-components: x86
//@[msvc64] compile-flags: --target x86_64-pc-windows-msvc
//@[msvc64] needs-llvm-components: x86

//@ add-core-stubs
#![feature(no_core)]
#![no_core]
extern crate minicore;
use minicore::*;

// Fits in i64 but not i32.
// C compiler demo: <https://godbolt.org/z/6v941G3x5>
// FIXME: This seems to be wrong for linux32?
#[repr(C)]
#[warn(overflowing_literals)]
enum OverflowingEnum {
    A = 9223372036854775807, // i64::MAX
    //[linux32,msvc32,msvc64]~^ WARN: literal out of range
}

#[cfg(not(linux64))]
const _: () = if mem::size_of::<OverflowingEnum>() != 4 {
    unsafe { hint::unreachable() }
};
#[cfg(linux64)]
const _: () = if mem::size_of::<OverflowingEnum>() != 8 {
    unsafe { hint::unreachable() }
};

fn main() {}
