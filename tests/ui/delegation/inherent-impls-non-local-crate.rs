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
//~^ ERROR: the placeholder `_` is not allowed within types on item signatures for functions
//~| ERROR: multiple applicable items in scope

reuse inherent_impl::X::<usize>::foo as x_foo_1;
reuse inherent_impl::X::<String>::foo as x_foo_2;

reuse inherent_impl::X::<_>::foo as x_foo_3;
//~^ ERROR: the placeholder `_` is not allowed within types on item signatures for functions
//~| ERROR: multiple applicable items in scope [E0034]

reuse inherent_impl::X::<()>::foo as x_foo_4;
//~^ ERROR: no associated function or constant named `foo` found for struct `X<()>` in the current scope


fn main() {}
