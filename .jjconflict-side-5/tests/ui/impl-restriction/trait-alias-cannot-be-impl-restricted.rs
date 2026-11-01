#![feature(impl_restriction, auto_traits, const_trait_impl, trait_alias)]

impl(crate) trait Alias = Copy; //~ ERROR trait aliases cannot be `impl`-restricted
impl(in crate) auto trait AutoAlias = Copy; //~ ERROR trait aliases cannot be `impl`-restricted
//~^ ERROR trait aliases cannot be `auto`
impl(self) unsafe trait UnsafeAlias = Copy; //~ ERROR trait aliases cannot be `impl`-restricted
//~^ ERROR trait aliases cannot be `unsafe`
impl(in self) const trait ConstAlias = Copy; //~ ERROR trait aliases cannot be `impl`-restricted

mod foo {
    impl(super) trait InnerAlias = Copy; //~ ERROR trait aliases cannot be `impl`-restricted
    impl(in crate::foo) const unsafe trait InnerConstUnsafeAlias = Copy; //~ ERROR trait aliases cannot be `impl`-restricted
    //~^ ERROR trait aliases cannot be `unsafe`
    impl(in crate::foo) unsafe auto trait InnerUnsafeAutoAlias = Copy; //~ ERROR trait aliases cannot be `impl`-restricted
    //~^ ERROR trait aliases cannot be `auto`
    //~^^ ERROR trait aliases cannot be `unsafe`
}

fn main() {}
