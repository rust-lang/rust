//@ aux-crate:inherent_impl=inherent-impl.rs

#![feature(fn_delegation)]

reuse inherent_impl::S::foo;

reuse inherent_impl::S::not_existing;
//~^ ERROR: no associated function or constant named `not_existing` found for struct `S` in the current scope
reuse inherent_impl::S::TYPE;
//~^ ERROR: no associated function or constant named `TYPE` found for struct `S` in the current scope
reuse inherent_impl::S::CONST;
//~^ ERROR: expected function, found `usize` [E0618]

reuse inherent_impl::S::bar;
//~^ ERROR: no associated function or constant named `bar` found for struct `S` in the current scope

reuse <inherent_impl::S as inherent_impl::Trait>::bar as trait_bar;

reuse inherent_impl::X::foo as x_foo;
//~^ ERROR: ambiguous delegation to inherent impl function
//~| ERROR: multiple applicable items in scope

fn main() {}
