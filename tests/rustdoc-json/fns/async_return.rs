//@ edition:2021

// Regression test for <https://github.com/rust-lang/rust/issues/101199>

use std::future::Future;

//@ is "$.types[0].primitive" \"i32\"

//@ is "$.index[?(@.name=='get_int')].inner.function.sig.output" 0
//@ is "$.index[?(@.name=='get_int')].inner.function.header.is_async" false
pub fn get_int() -> i32 {
    42
}

//@ is "$.index[?(@.name=='get_int_async')].inner.function.sig.output" 0
//@ is "$.index[?(@.name=='get_int_async')].inner.function.header.is_async" true
pub async fn get_int_async() -> i32 {
    42
}

//@ is "$.index[?(@.name=='get_int_future')].inner.function.sig.output" 1
//@ is "$.types[1].impl_trait[0].trait_bound.trait.path" '"Future"'
//@ is "$.types[1].impl_trait[0].trait_bound.trait.args.angle_bracketed.constraints[0].name" '"Output"'
//@ is "$.types[1].impl_trait[0].trait_bound.trait.args.angle_bracketed.constraints[0].binding.equality.type" 0
//@ is "$.index[?(@.name=='get_int_future')].inner.function.header.is_async" false
pub fn get_int_future() -> impl Future<Output = i32> {
    async { 42 }
}

//@ is "$.index[?(@.name=='get_int_future_async')].inner.function.sig.output" 1
//@ is "$.index[?(@.name=='get_int_future_async')].inner.function.header.is_async" true
pub async fn get_int_future_async() -> impl Future<Output = i32> {
    async { 42 }
}
