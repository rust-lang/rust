//@ compile-flags: -Znext-solver
#![cfg_attr(any(yyy, yyn, yny, ynn), feature(const_trait_impl))]

//@ revisions: yyy yyn yny ynn nyy nyn nny nnn
//@[yyy] check-pass
/// yyy: feature enabled, Foo is const, Bar is const
/// yyn: feature enabled, Foo is const, Bar is not const
/// yny: feature enabled, Foo is not const, Bar is const
/// ynn: feature enabled, Foo is not const, Bar is not const
/// nyy: feature not enabled, Foo is const, Bar is const
/// nyn: feature not enabled, Foo is const, Bar is not const
/// nny: feature not enabled, Foo is not const, Bar is const
/// nnn: feature not enabled, Foo is not const, Bar is not const

#[cfg_attr(any(yyy, yyn, nyy, nyn), const_trait)]
//[nyy,nyn]~^ ERROR: `const_trait` is a temporary placeholder for marking a trait that is suitable for `const` `impls` and all default bodies as `const`, which may be removed or renamed in the future
trait Foo {
    fn a(&self);
}

#[cfg_attr(any(yyy, yny, nyy, nyn), const_trait)]
//[nyy,nyn]~^ ERROR: `const_trait` is a temporary placeholder for marking a trait that is suitable for `const` `impls` and all default bodies as `const`, which may be removed or renamed in the future
trait Bar: [const] Foo {}
//[yny,ynn,nny,nnn]~^ ERROR: `[const]` can only be applied to `const` traits
//[yny,ynn,nny,nnn]~| ERROR: `[const]` can only be applied to `const` traits
//[yny,ynn,nny,nnn]~| ERROR: `[const]` can only be applied to `const` traits
//[yny]~^^^^ ERROR: `[const]` can only be applied to `const` traits
//[yny]~| ERROR: `[const]` can only be applied to `const` traits
//[yyn,ynn,nny,nnn]~^^^^^^ ERROR: `[const]` is not allowed here
//[nyy,nyn,nny,nnn]~^^^^^^^ ERROR: const trait impls are experimental

const fn foo<T: [const] Bar>(x: &T) {
    //[yyn,ynn,nny,nnn]~^ ERROR: `[const]` can only be applied to `const` traits
    //[yyn,ynn,nny,nnn]~| ERROR: `[const]` can only be applied to `const` traits
    //[nyy,nyn,nny,nnn]~^^^ ERROR: const trait impls are experimental
    x.a();
    //[yyn]~^ ERROR: the trait bound `T: [const] Foo` is not satisfied
    //[ynn,yny,nny,nnn]~^^ ERROR: cannot call non-const method `<T as Foo>::a` in constant functions
    //[nyy,nyn]~^^^ ERROR: cannot call conditionally-const method `<T as Foo>::a` in constant functions
}

fn main() {}
