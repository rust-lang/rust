// Regression test for issue #116629.
// Check that we don't render host effect parameters & arguments.

//@ aux-crate:const_effect_param=const-effect-param.rs
//@ edition: 2021
#![crate_name = "user"]

// Don't render the host param on `load` and the host arg `host` passed to `Resource`.
//@ has user/fn.load.html
//@ has - '//pre[@class="rust item-decl"]' "pub const fn load<R>() -> i32\
//     where \
//         R: Resource"
pub use const_effect_param::load;

// Don't render the host arg `true` passed to `Resource`.
//@ has user/fn.lock.html
//@ has - '//pre[@class="rust item-decl"]' "pub const fn lock<R>()\
//     where \
//         R: Resource"
pub use const_effect_param::lock;

// Regression test for an issue introduced in PR #116670.
// Don't hide the const param `host` since it actually isn't the host effect param.
//@ has user/fn.clash.html
//@ has - '//pre[@class="rust item-decl"]' \
//    "pub const fn clash<T, const host: u64>()\
//     where \
//         T: Clash<host>"
pub use const_effect_param::clash;
