//@ known-bug: #110395
//@ failure-status: 101
//@ dont-check-compiler-stderr
// FIXME(effects) check-pass
//@ compile-flags: -Znext-solver

#![crate_type = "lib"]
#![allow(internal_features, incomplete_features)]
#![no_std]
#![no_core]
#![feature(
    auto_traits,
    const_trait_impl,
    effects,
    lang_items,
    no_core,
    staged_api,
    unboxed_closures,
    rustc_attrs,
    marker_trait_attr,
)]
#![stable(feature = "minicore", since = "1.0.0")]

fn test() {
    fn is_const_fn<F>(_: F)
    where
        F: const FnOnce<()>,
    {
    }

    const fn foo() {}

    is_const_fn(foo);
}

/// ---------------------------------------------------------------------- ///
/// Const fn trait definitions

#[const_trait]
#[lang = "fn"]
#[rustc_paren_sugar]
trait Fn<Args: Tuple>: ~const FnMut<Args> {
    extern "rust-call" fn call(&self, args: Args) -> Self::Output;
}

#[const_trait]
#[lang = "fn_mut"]
#[rustc_paren_sugar]
trait FnMut<Args: Tuple>: ~const FnOnce<Args> {
    extern "rust-call" fn call_mut(&mut self, args: Args) -> Self::Output;
}

#[const_trait]
#[lang = "fn_once"]
#[rustc_paren_sugar]
trait FnOnce<Args: Tuple> {
    #[lang = "fn_once_output"]
    type Output;

    extern "rust-call" fn call_once(self, args: Args) -> Self::Output;
}

/// ---------------------------------------------------------------------- ///
/// All this other stuff needed for core. Unrelated to test.

#[lang = "destruct"]
#[const_trait]
trait Destruct {}

#[lang = "freeze"]
unsafe auto trait Freeze {}

#[lang = "drop"]
#[const_trait]
trait Drop {
    fn drop(&mut self);
}

#[lang = "sized"]
trait Sized {}
#[lang = "copy"]
trait Copy {}

#[lang = "tuple_trait"]
trait Tuple {}

#[lang = "legacy_receiver"]
trait LegacyReceiver {}

impl<T: ?Sized> LegacyReceiver for &T {}
