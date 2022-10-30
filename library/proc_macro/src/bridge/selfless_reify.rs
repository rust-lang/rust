//! Abstraction for creating `fn` pointers from any callable that *effectively*
//! has the equivalent of implementing `Default`, even if the compiler neither
//! provides `Default` nor allows reifying closures (i.e. creating `fn` pointers)
//! other than those with absolutely no captures.
//!
//! More specifically, for a closure-like type to be "effectively `Default`":
//! * it must be a ZST (zero-sized type): no information contained within, so
//!   that `Default`'s return value (if it were implemented) is unambiguous
//! * it must be `Copy`: no captured "unique ZST tokens" or any other similar
//!   types that would make duplicating values at will unsound
//!   * combined with the ZST requirement, this confers a kind of "telecopy"
//!     ability: similar to `Copy`, but without keeping the value around, and
//!     instead "reconstructing" it (a noop given it's a ZST) when needed
//! * it must be *provably* inhabited: no captured uninhabited types or any
//!   other types that cannot be constructed by the user of this abstraction
//!   * the proof is a value of the closure-like type itself, in a sense the
//!     "seed" for the "telecopy" process made possible by ZST + `Copy`
//!   * this requirement is the only reason an abstraction limited to a specific
//!     usecase is required: ZST + `Copy` can be checked with *at worst* a panic
//!     at the "attempted `::default()` call" time, but that doesn't guarantee
//!     that the value can be soundly created, and attempting to use the typical
//!     "proof ZST token" approach leads yet again to having a ZST + `Copy` type
//!     that is not proof of anything without a value (i.e. isomorphic to a
//!     newtype of the type it's trying to prove the inhabitation of)
//!
//! A more flexible (and safer) solution to the general problem could exist once
//! `const`-generic parameters can have type parameters in their types:
//!
//! ```rust,ignore (needs future const-generics)
//! extern "C" fn ffi_wrapper<
//!     A, R,
//!     F: Fn(A) -> R,
//!     const f: F, // <-- this `const`-generic is not yet allowed
//! >(arg: A) -> R {
//!     f(arg)
//! }
//! ```

use std::mem;

// FIXME(eddyb) this could be `trait` impls except for the `const fn` requirement.
macro_rules! define_reify_functions {
    ($(
        fn $name:ident $(<$($param:ident),*>)?
            for $(extern $abi:tt)? fn($($arg:ident: $arg_ty:ty),*) -> $ret_ty:ty;
    )+) => {
        $(pub const fn $name<
            $($($param,)*)?
            F: Fn($($arg_ty),*) -> $ret_ty + Copy
        >(f: F) -> $(extern $abi)? fn($($arg_ty),*) -> $ret_ty {
            // FIXME(eddyb) describe the `F` type (e.g. via `type_name::<F>`) once panic
            // formatting becomes possible in `const fn`.
            assert!(mem::size_of::<F>() == 0, "selfless_reify: closure must be zero-sized");

            $(extern $abi)? fn wrapper<
                $($($param,)*)?
                F: Fn($($arg_ty),*) -> $ret_ty + Copy
            >($($arg: $arg_ty),*) -> $ret_ty {
                let f = unsafe {
                    // SAFETY: `F` satisfies all criteria for "out of thin air"
                    // reconstructability (see module-level doc comment).
                    mem::MaybeUninit::<F>::uninit().assume_init()
                };
                f($($arg),*)
            }
            let _f_proof = f;
            wrapper::<
                $($($param,)*)?
                F
            >
        })+
    }
}

define_reify_functions! {
    fn _reify_to_extern_c_fn_unary<A, R> for extern "C" fn(arg: A) -> R;

    // HACK(eddyb) this abstraction is used with `for<'a> fn(BridgeConfig<'a>)
    // -> T` but that doesn't work with just `reify_to_extern_c_fn_unary`
    // because of the `fn` pointer type being "higher-ranked" (i.e. the
    // `for<'a>` binder).
    // FIXME(eddyb) try to remove the lifetime from `BridgeConfig`, that'd help.
    fn reify_to_extern_c_fn_hrt_bridge<R> for extern "C" fn(bridge: super::BridgeConfig<'_>) -> R;
}
