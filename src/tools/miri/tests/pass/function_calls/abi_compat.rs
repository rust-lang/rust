#![feature(never_type)]

use std::rc::Rc;
use std::{mem, num, ptr};

#[derive(Copy, Clone, Default)]
struct Zst;

#[repr(transparent)]
#[derive(Copy, Clone)]
struct Wrapper<T>(T);

fn id<T>(x: T) -> T {
    x
}

#[derive(Copy, Clone)]
enum Either<T, U> {
    Left(T),
    Right(U),
}
#[derive(Copy, Clone)]
enum Either2<T, U> {
    Left(T),
    #[allow(unused)]
    Right(U, ()),
}

fn test_abi_compat<T: Clone, U: Clone>(t: T, u: U) {
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
    let _val = f(u.clone());
    let f: fn(U) -> U = id;
    let f: fn(T) -> T = unsafe { std::mem::transmute(f) };
    let _val = f(t.clone());

    // And then we do the same for `extern "C"`.
    let f: extern "C" fn(T) -> T = id_c;
    let f: extern "C" fn(U) -> U = unsafe { std::mem::transmute(f) };
    let _val = f(u);
    let f: extern "C" fn(U) -> U = id_c;
    let f: extern "C" fn(T) -> T = unsafe { std::mem::transmute(f) };
    let _val = f(t);
}

/// Ensure that `T` is compatible with various repr(transparent) wrappers around `T`.
fn test_abi_newtype<T: Copy + Default>() {
    #[repr(transparent)]
    #[derive(Copy, Clone)]
    struct Wrapper2<T>(T, ());
    #[repr(transparent)]
    #[derive(Copy, Clone)]
    struct Wrapper2a<T>((), T);
    #[repr(transparent)]
    #[derive(Copy, Clone)]
    struct Wrapper3<T>(Zst, T, [u8; 0]);

    let t = T::default();
    test_abi_compat(t, Wrapper(t));
    test_abi_compat(t, Wrapper2(t, ()));
    test_abi_compat(t, Wrapper2a((), t));
    test_abi_compat(t, Wrapper3(Zst, t, []));
    test_abi_compat(t, mem::MaybeUninit::new(t)); // MaybeUninit is `repr(transparent)`
}

fn main() {
    // Here we check some of the guaranteed ABI compatibilities:
    // - Different integer types of the same size and sign.
    if cfg!(target_pointer_width = "32") {
        test_abi_compat(0usize, 0u32);
        test_abi_compat(0isize, 0i32);
    } else {
        test_abi_compat(0usize, 0u64);
        test_abi_compat(0isize, 0i64);
    }
    test_abi_compat(42u32, num::NonZero::new(1u32).unwrap());
    // - `char` and `u32`.
    test_abi_compat(42u32, 'x');
    // - Reference/pointer types with the same pointee.
    test_abi_compat(&0u32, &0u32 as *const u32);
    test_abi_compat(&mut 0u32 as *mut u32, Box::new(0u32));
    test_abi_compat(&(), ptr::NonNull::<()>::dangling());
    // - Reference/pointer types with different but sized pointees.
    test_abi_compat(&0u32, &([true; 4], [0u32; 0]));
    // - `fn` types
    test_abi_compat(main as fn(), id::<i32> as fn(i32) -> i32);
    // - 1-ZST
    test_abi_compat((), [0u8; 0]);

    // Guaranteed null-pointer-layout optimizations:
    // - Guaranteed Option<X> null-pointer-optimizations (RFC 3391).
    test_abi_compat(&0u32 as *const u32, Some(&0u32));
    test_abi_compat(main as fn(), Some(main as fn()));
    test_abi_compat(0u32, Some(num::NonZero::new(1u32).unwrap()));
    test_abi_compat(&0u32 as *const u32, Some(Wrapper(&0u32)));
    test_abi_compat(0u32, Some(Wrapper(num::NonZeroU32::new(1u32).unwrap())));
    // - Guaranteed Result<X, ZST1> does the same as Option<X> (RFC 3391)
    test_abi_compat(&0u32 as *const u32, Result::<_, ()>::Ok(&0u32));
    test_abi_compat(&0u32 as *const u32, Result::<_, !>::Ok(&0u32));
    test_abi_compat(main as fn(), Result::<_, ()>::Ok(main as fn()));
    test_abi_compat(0u32, Result::<_, ()>::Ok(num::NonZeroU32::new(1).unwrap()));
    test_abi_compat(&0u32 as *const u32, Result::<_, ()>::Ok(Wrapper(&0u32)));
    test_abi_compat(0u32, Result::<_, ()>::Ok(Wrapper(num::NonZeroU32::new(1).unwrap())));
    // - Guaranteed Result<ZST1, X> also does the same as Option<X> (RFC 3391)
    test_abi_compat(&0u32 as *const u32, Result::<(), _>::Err(&0u32));
    test_abi_compat(main as fn(), Result::<(), _>::Err(main as fn()));
    test_abi_compat(0u32, Result::<(), _>::Err(num::NonZeroU32::new(1).unwrap()));
    test_abi_compat(&0u32 as *const u32, Result::<(), _>::Err(Wrapper(&0u32)));
    test_abi_compat(0u32, Result::<(), _>::Err(Wrapper(num::NonZeroU32::new(1).unwrap())));
    // - Guaranteed null-pointer-optimizations for custom option-like types
    test_abi_compat(&0u32 as *const u32, Either::<_, ()>::Left(&0u32));
    test_abi_compat(&0u32 as *const u32, Either::<_, !>::Left(&0u32));
    test_abi_compat(&0u32 as *const u32, Either::<(), _>::Right(&0u32));
    test_abi_compat(&0u32 as *const u32, Either::<!, _>::Right(&0u32));
    test_abi_compat(&0u32 as *const u32, Either2::<_, ()>::Left(&0u32));
    test_abi_compat(&0u32 as *const u32, Either2::<_, [u8; 0]>::Left(&0u32));

    // These must work for *any* type, since we guarantee that `repr(transparent)` is ABI-compatible
    // with the wrapped field.
    test_abi_newtype::<()>();
    test_abi_newtype::<Zst>();
    test_abi_newtype::<u32>();
    test_abi_newtype::<f32>();
    test_abi_newtype::<(u8, u16, f32)>();
    test_abi_newtype::<[u8; 0]>();
    test_abi_newtype::<[u32; 0]>();
    test_abi_newtype::<[u32; 2]>();
    test_abi_newtype::<[u32; 32]>();
    test_abi_newtype::<Option<i32>>();
    test_abi_newtype::<Option<num::NonZero<u32>>>();

    // Extra test for assumptions made by arbitrary-self-dyn-receivers.
    // This is interesting since these types are not `repr(transparent)`. So this is not part of our
    // public ABI guarantees, but is relied on by the compiler.
    let rc = Rc::new(0);
    let rc_ptr: *mut i32 = unsafe { mem::transmute_copy(&rc) };
    test_abi_compat(rc, rc_ptr);
}
