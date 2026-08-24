//@ aux-crate:inherent_impl=inherent-impl.rs

#![feature(fn_delegation)]

reuse inherent_impl::S::foo;
//~^ ERROR: cannot find function `foo` in `inherent_impl::S`

reuse inherent_impl::S::not_existing;
//~^ ERROR: cannot find function `not_existing` in `inherent_impl::S`

reuse inherent_impl::S::TYPE;
//~^ ERROR: cannot find function `TYPE` in `inherent_impl::S`

reuse inherent_impl::S::CONST;
//~^ ERROR: cannot find function `CONST` in `inherent_impl::S`

reuse inherent_impl::S::bar;
//~^ ERROR: cannot find function `bar` in `inherent_impl::S`

reuse <inherent_impl::S as inherent_impl::Trait>::bar as trait_bar;

reuse inherent_impl::X::foo as x_foo;
//~^ ERROR: cannot find function `foo` in `inherent_impl::X`

fn main() {}
