// Regression test for issue #116629.
// Check that we render the correct generic params of const fn

// aux-crate:const_fn=const-fn.rs
// edition: 2021
#![crate_name = "user"]

// @has user/fn.load.html
// @has - '//pre[@class="rust item-decl"]' "pub const fn load() -> i32"
pub use const_fn::load;
