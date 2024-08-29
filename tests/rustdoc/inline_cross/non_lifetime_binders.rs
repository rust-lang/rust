//@ aux-crate:non_lifetime_binders=non_lifetime_binders.rs
//@ edition: 2021
#![crate_name = "user"]

//@ has user/fn.f.html
//@ has - '//pre[@class="rust item-decl"]' "f(_: impl for<T> Trait<T>)"
pub use non_lifetime_binders::f;

//@ has user/fn.g.html
//@ has - '//pre[@class="rust item-decl"]' "g<T>(_: T)\
// where \
//     T: for<U> Trait<U>"
pub use non_lifetime_binders::g;
