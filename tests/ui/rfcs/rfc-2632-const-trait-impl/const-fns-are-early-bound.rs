//@ known-bug: #110395
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

impl<T: ?Sized> LegacyReceiver for &mut T {}

#[stable(feature = "minicore", since = "1.0.0")]
pub mod effects {
    use super::Sized;

    #[lang = "EffectsNoRuntime"]
    #[stable(feature = "minicore", since = "1.0.0")]
    pub struct NoRuntime;
    #[lang = "EffectsMaybe"]
    #[stable(feature = "minicore", since = "1.0.0")]
    pub struct Maybe;
    #[lang = "EffectsRuntime"]
    #[stable(feature = "minicore", since = "1.0.0")]
    pub struct Runtime;

    #[lang = "EffectsCompat"]
    #[stable(feature = "minicore", since = "1.0.0")]
    pub trait Compat<#[rustc_runtime] const RUNTIME: bool> {}

    #[stable(feature = "minicore", since = "1.0.0")]
    impl Compat<false> for NoRuntime {}
    #[stable(feature = "minicore", since = "1.0.0")]
    impl Compat<true> for Runtime {}
    #[stable(feature = "minicore", since = "1.0.0")]
    impl<#[rustc_runtime] const RUNTIME: bool> Compat<RUNTIME> for Maybe {}

    #[lang = "EffectsTyCompat"]
    #[marker]
    #[stable(feature = "minicore", since = "1.0.0")]
    pub trait TyCompat<T: ?Sized> {}

    #[stable(feature = "minicore", since = "1.0.0")]
    impl<T: ?Sized> TyCompat<T> for T {}
    #[stable(feature = "minicore", since = "1.0.0")]
    impl<T: ?Sized> TyCompat<T> for Maybe {}
    #[stable(feature = "minicore", since = "1.0.0")]
    impl<T: ?Sized> TyCompat<Maybe> for T {}

    #[lang = "EffectsIntersection"]
    #[stable(feature = "minicore", since = "1.0.0")]
    pub trait Intersection {
        #[lang = "EffectsIntersectionOutput"]
        #[stable(feature = "minicore", since = "1.0.0")]
        type Output: ?Sized;
    }
}
