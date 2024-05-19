//@ edition:2021
// ignore-tidy-linelength

// Regression test for <https://github.com/rust-lang/rust/issues/101199>

use std::future::Future;

// @is "$.index[*][?(@.name=='get_int')].inner.function.decl.output.primitive" \"i32\"
// @is "$.index[*][?(@.name=='get_int')].inner.function.header.async" false
pub fn get_int() -> i32 {
    42
}

// @is "$.index[*][?(@.name=='get_int_async')].inner.function.decl.output.primitive" \"i32\"
// @is "$.index[*][?(@.name=='get_int_async')].inner.function.header.async" true
pub async fn get_int_async() -> i32 {
    42
}

// @is "$.index[*][?(@.name=='get_int_future')].inner.function.decl.output.impl_trait[0].trait_bound.trait.name" '"Future"'
// @is "$.index[*][?(@.name=='get_int_future')].inner.function.decl.output.impl_trait[0].trait_bound.trait.args.angle_bracketed.bindings[0].name" '"Output"'
// @is "$.index[*][?(@.name=='get_int_future')].inner.function.decl.output.impl_trait[0].trait_bound.trait.args.angle_bracketed.bindings[0].binding.equality.type.primitive"  \"i32\"
// @is "$.index[*][?(@.name=='get_int_future')].inner.function.header.async" false
pub fn get_int_future() -> impl Future<Output = i32> {
    async { 42 }
}

// @is "$.index[*][?(@.name=='get_int_future_async')].inner.function.decl.output.impl_trait[0].trait_bound.trait.name" '"Future"'
// @is "$.index[*][?(@.name=='get_int_future_async')].inner.function.decl.output.impl_trait[0].trait_bound.trait.args.angle_bracketed.bindings[0].name" '"Output"'
// @is "$.index[*][?(@.name=='get_int_future_async')].inner.function.decl.output.impl_trait[0].trait_bound.trait.args.angle_bracketed.bindings[0].binding.equality.type.primitive" \"i32\"
// @is "$.index[*][?(@.name=='get_int_future_async')].inner.function.header.async" true
pub async fn get_int_future_async() -> impl Future<Output = i32> {
    async { 42 }
}
