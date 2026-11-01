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

#[cfg(any(yyy, yyn, nyy, nyn))] const trait Foo { fn a(&self); }
//[nyy,nyn,nny,nnn]~^ ERROR: const trait impls are experimental
#[cfg(any(yny, ynn, nny, nnn))] trait Foo { fn a(&self); }

#[cfg(any(yyy, yny, nyy, nny))] const trait Bar: [const] Foo {}
//[nyy,nyn,nny,nnn]~^ ERROR: const trait impls are experimental
//[nyy,nyn,nny,nnn]~| ERROR: const trait impls are experimental
//[yny,nny]~^^^ ERROR: `[const]` can only be applied to `const` traits
//[yny,nny]~| ERROR: `[const]` can only be applied to `const` traits
//[yny,nny]~| ERROR: `[const]` can only be applied to `const` traits
//[yny,nny]~| ERROR: `[const]` can only be applied to `const` traits
//[yny,nny]~| ERROR: `[const]` can only be applied to `const` traits
#[cfg(any(yyn, ynn, nyn, nnn))] trait Bar: [const] Foo {}
//[yyn,ynn,nyn,nnn]~^ ERROR: `[const]` is not allowed here
//[nyy,nyn,nny,nnn]~^^ ERROR: const trait impls are experimental
//[ynn,nnn]~^^^ ERROR: `[const]` can only be applied to `const` traits
//[ynn,nnn]~| ERROR: `[const]` can only be applied to `const` traits
//[ynn,nnn]~| ERROR: `[const]` can only be applied to `const` traits

const fn foo<T: [const] Bar>(x: &T) {
    //[yyn,ynn,nyn,nnn]~^ ERROR: `[const]` can only be applied to `const` traits
    //[yyn,ynn,nyn,nnn]~| ERROR: `[const]` can only be applied to `const` traits
    //[nyy,nyn,nny,nnn]~^^^ ERROR: const trait impls are experimental
    x.a();
    //[yyn]~^ ERROR: the trait bound `T: [const] Foo` is not satisfied
    //[ynn,yny,nny,nnn,nyn]~^^ ERROR: cannot call non-const method `<T as Foo>::a` in constant functions
    //[nyy]~^^^ ERROR: cannot call conditionally-const method `<T as Foo>::a` in constant functions
}

fn main() {}
