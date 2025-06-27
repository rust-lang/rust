//@ edition:2021

// Regression test for <https://github.com/rust-lang/rust/issues/101199>

use std::future::Future;

//@ jq .index[] | select(.name == "get_int").inner.function | .sig?.output.primitive == "i32" and .header?.is_async == false
pub fn get_int() -> i32 {
    42
}

//@ jq .index[] | select(.name == "get_int_async").inner.function | .sig?.output.primitive == "i32" and .header?.is_async == true
pub async fn get_int_async() -> i32 {
    42
}

//@ arg get_int_future .index[] | select(.name == "get_int_future").inner.function
//@ jq $get_int_future.sig?.output.impl_trait[]?.trait_bound.trait?.path == "Future"
//@ jq $get_int_future.sig?.output.impl_trait[]?.trait_bound.trait?.args.angle_bracketed?.constraints?[].name == "Output"
//@ jq $get_int_future.sig?.output.impl_trait[]?.trait_bound.trait?.args.angle_bracketed?.constraints?[].binding.equality.type?.primitive == "i32"
//@ jq $get_int_future.header?.is_async == false
pub fn get_int_future() -> impl Future<Output = i32> {
    async { 42 }
}

//@ arg get_int_future_async .index[] | select(.name == "get_int_future_async").inner.function
//@ jq $get_int_future_async.sig?.output.impl_trait[]?.trait_bound.trait?.path == "Future"
//@ jq $get_int_future_async.sig?.output.impl_trait[]?.trait_bound.trait?.args.angle_bracketed?.constraints?[].name == "Output"
//@ jq $get_int_future_async.sig?.output.impl_trait[]?.trait_bound.trait?.args.angle_bracketed?.constraints?[].binding.equality.type?.primitive == "i32"
//@ jq $get_int_future_async.header?.is_async == true
pub async fn get_int_future_async() -> impl Future<Output = i32> {
    async { 42 }
}
