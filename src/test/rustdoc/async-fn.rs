// edition:2018
// compile-flags:-Z unstable-options

// FIXME: once `--edition` is stable in rustdoc, remove that `compile-flags` directive

#![feature(async_await, futures_api)]

// @has async_fn/struct.S.html
// @has - '//code' 'pub async fn f()'
pub struct S;

impl S {
    pub async fn f() {}
}
