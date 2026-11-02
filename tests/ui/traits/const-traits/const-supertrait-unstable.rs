//! Test output of const super trait errors in both stable and nightly.
//! We don't want to provide suggestions on stable that only make sense in nightly.

//@ needs-target-std
//@ ignore-backends: gcc

//@ normalize-stderr: "dev release channel" -> "nightly release channel"
//@ normalize-stderr: "stable release channel" -> "nightly release channel"
//@ compile-flags: --check-cfg=cfg(feature_enabled)

//@ revisions: stable stable-enabled nightly nightly-enabled
//@[stable] act-as-stable
//@[stable-enabled] act-as-stable
//@[stable-enabled] compile-flags: --cfg feature_enabled
//@[nightly] only-nightly
//@[nightly-enabled] only-nightly
//@[nightly-enabled] compile-flags: --cfg feature_enabled

#![cfg_attr(feature_enabled, feature(const_trait_impl))]
//[stable-enabled]~^ ERROR: `#![feature]` may not be used

trait Foo {
    fn a(&self);
}

trait Bar: [const] Foo {}
//~^ ERROR: `[const]` is not allowed here
//[nightly,stable]~^^ ERROR: const trait impls are experimental [E0658]
//~| ERROR: `[const]` can only be applied to `const` traits
//~| ERROR: `[const]` can only be applied to `const` traits
//~| ERROR: `[const]` can only be applied to `const` traits

const fn foo<T: [const] Bar>(x: &T) {
    //[nightly,stable]~^ ERROR: const trait impls are experimental [E0658]
    //~^^ ERROR: `[const]` can only be applied to `const` traits
    //~| ERROR: `[const]` can only be applied to `const` traits
    x.a();
    //~^ ERROR: cannot call non-const method `<T as Foo>::a` in constant functions [E0015]
}

fn main() {}
