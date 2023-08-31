#![feature(portable_simd)]
use std::mem;
use std::num;
use std::simd;

#[derive(Copy, Clone)]
struct Zst;

fn test_abi_compat<T: Copy, U: Copy>(t: T, u: U) {
    fn id<T>(x: T) -> T {
        x
    }
    extern "C" fn id_c<T>(x: T) -> T {
        x
    }

    // This checks ABI compatibility both for arguments and return values,
    // in both directions.
    let f: fn(T) -> T = id;
    let f: fn(U) -> U = unsafe { std::mem::transmute(f) };
    let _val = f(u);
    let f: fn(U) -> U = id;
    let f: fn(T) -> T = unsafe { std::mem::transmute(f) };
    let _val = f(t);

    // And then we do the same for `extern "C"`.
    let f: extern "C" fn(T) -> T = id_c;
    let f: extern "C" fn(U) -> U = unsafe { std::mem::transmute(f) };
    let _val = f(u);
    let f: extern "C" fn(U) -> U = id_c;
    let f: extern "C" fn(T) -> T = unsafe { std::mem::transmute(f) };
    let _val = f(t);
}

/// Ensure that `T` is compatible with various repr(transparent) wrappers around `T`.
fn test_abi_newtype<T: Copy>(t: T) {
    #[repr(transparent)]
    #[derive(Copy, Clone)]
    struct Wrapper1<T>(T);
    #[repr(transparent)]
    #[derive(Copy, Clone)]
    struct Wrapper2<T>(T, ());
    #[repr(transparent)]
    #[derive(Copy, Clone)]
    struct Wrapper2a<T>((), T);
    #[repr(transparent)]
    #[derive(Copy, Clone)]
    struct Wrapper3<T>(Zst, T, [u8; 0]);

    test_abi_compat(t, Wrapper1(t));
    test_abi_compat(t, Wrapper2(t, ()));
    test_abi_compat(t, Wrapper2a((), t));
    test_abi_compat(t, Wrapper3(Zst, t, []));
    test_abi_compat(t, mem::MaybeUninit::new(t)); // MaybeUninit is `repr(transparent)`
}

fn main() {
    // Here we check:
    // - unsigned vs signed integer is allowed
    // - u32/i32 vs char is allowed
    // - u32 vs NonZeroU32/Option<NonZeroU32> is allowed
    // - reference vs raw pointer is allowed
    // - references to things of the same size and alignment are allowed
    // These are very basic tests that should work on all ABIs. However it is not clear that any of
    // these would be stably guaranteed. Code that relies on this is equivalent to code that relies
    // on the layout of `repr(Rust)` types. They are also fragile: the same mismatches in the fields
    // of a struct (even with `repr(C)`) will not always be accepted by Miri.
    test_abi_compat(0u32, 0i32);
    test_abi_compat(simd::u32x8::splat(1), simd::i32x8::splat(1));
    test_abi_compat(0u32, 'x');
    test_abi_compat(0i32, 'x');
    test_abi_compat(42u32, num::NonZeroU32::new(1).unwrap());
    test_abi_compat(0u32, Some(num::NonZeroU32::new(1).unwrap()));
    test_abi_compat(&0u32, &0u32 as *const u32);
    test_abi_compat(&0u32, &([true; 4], [0u32; 0]));
    // Note that `bool` and `u8` are *not* compatible, at least on x86-64!
    // One of them has `arg_ext: Zext`, the other does not.

    // These must work for *any* type, since we guarantee that `repr(transparent)` is ABI-compatible
    // with the wrapped field.
    test_abi_newtype(());
    // FIXME: this still fails! test_abi_newtype(Zst);
    test_abi_newtype(0u32);
    test_abi_newtype(0f32);
    test_abi_newtype((0u32, 1u32, 2u32));
    test_abi_newtype([0u32, 1u32, 2u32]);
    test_abi_newtype([0i32; 0]);
}
