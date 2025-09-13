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

// Each value fits in i32 or u32, but not all values fit into the same type.
// C compiler demo: <https://godbolt.org/z/GPYafPhjK>
// FIXME: This seems to do the wrong thing for 32bit Linux?
#[repr(C)]
#[warn(overflowing_literals)]
enum OverflowingEnum2 {
    A = 4294967294, // u32::MAX - 1
    //[linux32,msvc32,msvc64]~^ WARN: literal out of range
    B = -1,
}

#[cfg(not(linux64))]
const _: () = if mem::size_of::<OverflowingEnum2>() != 4 {
    unsafe { hint::unreachable() }
};
#[cfg(linux64)]
const _: () = if mem::size_of::<OverflowingEnum2>() != 8 {
    unsafe { hint::unreachable() }
};

// Force i32 or u32, respectively.
// C compiler demo: <https://godbolt.org/z/bGss855a4>
#[repr(C)]
enum I32Enum {
    A = 2147483647, // i32::MAX
    B = -2147483647,
}
const _: () = if mem::size_of::<I32Enum>() != 4 {
    unsafe { hint::unreachable() }
};

// C compiler demo: <https://godbolt.org/z/MeE9YGj4n>
#[repr(C)]
#[allow(overflowing_literals)]
enum U32Enum {
    A = 4294967295, // u32::MAX
    B = 0,
}
const _: () = if mem::size_of::<U32Enum>() != 4 {
    unsafe { hint::unreachable() }
};

fn main() {}
