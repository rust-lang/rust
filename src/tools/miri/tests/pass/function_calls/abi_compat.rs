#![feature(portable_simd)]
use std::num;
use std::mem;
use std::simd;

fn test_abi_compat<T, U>(t: T, u: U) {
    fn id<T>(x: T) -> T { x }
    
    // This checks ABI compatibility both for arguments and return values,
    // in both directions.
    let f: fn(T) -> T = id;
    let f: fn(U) -> U = unsafe { std::mem::transmute(f) };
    drop(f(u));
    
    let f: fn(U) -> U = id;
    let f: fn(T) -> T = unsafe { std::mem::transmute(f) };
    drop(f(t));
}

/// Ensure that `T` is compatible with various repr(transparent) wrappers around `T`.
fn test_abi_newtype<T: Copy>(t: T) {
    #[repr(transparent)]
    struct Wrapper1<T>(T);
    #[repr(transparent)]
    struct Wrapper2<T>(T, ());
    #[repr(transparent)]
    struct Wrapper2a<T>((), T);
    #[repr(transparent)]
    struct Wrapper3<T>(T, [u8; 0]);

    test_abi_compat(t, Wrapper1(t));
    test_abi_compat(t, Wrapper2(t, ()));
    test_abi_compat(t, Wrapper2a((), t));
    test_abi_compat(t, Wrapper3(t, []));
}

fn main() {
    test_abi_compat(0u32, 'x');
    test_abi_compat(&0u32, &([true; 4], [0u32; 0]));
    test_abi_compat(0u32, mem::MaybeUninit::new(0u32));
    test_abi_compat(42u32, num::NonZeroU32::new(1).unwrap());
    test_abi_compat(0u32, Some(num::NonZeroU32::new(1).unwrap()));
    test_abi_compat(0u32, 0i32);
    test_abi_compat(simd::u32x8::splat(1), simd::i32x8::splat(1));
    // Note that `bool` and `u8` are *not* compatible, at least on x86-64!
    // One of them has `arg_ext: Zext`, the other does not.

    test_abi_newtype(0u32);
    test_abi_newtype(0f32);
    test_abi_newtype((0u32, 1u32, 2u32));
    test_abi_newtype([0u32, 1u32, 2u32]);
    test_abi_newtype([0i32; 0]);
}
