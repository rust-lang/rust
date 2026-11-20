// Regression test for issue #115760.
// Check that we render the correct return type of free and
// associated async functions reexported from external crates.

//@ aux-crate:async_fn=async-fn.rs
//@ edition: 2021
#![crate_name = "user"]

//@ has user/fn.load.html
//@ has - '//pre[@class="rust item-decl"]' "pub async fn load() -> i32"
pub use async_fn::load;

//@ has user/trait.Load.html
//@ has - '//*[@id="tymethod.run"]' 'async fn run(&self) -> i32'
pub use async_fn::Load;

//@ has user/struct.Loader.html
//@ has - '//*[@id="method.run"]' 'async fn run(&self) -> i32'
pub use async_fn::Loader;
