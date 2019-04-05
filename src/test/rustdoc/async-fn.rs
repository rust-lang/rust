// edition:2018

#![feature(async_await)]

// @has async_fn/fn.foo.html '//pre[@class="rust fn"]' 'pub async fn foo() -> Option<Foo>'
pub async fn foo() -> Option<Foo> {
    None
}

// @has async_fn/fn.bar.html '//pre[@class="rust fn"]' 'pub async fn bar(a: i32, b: i32) -> i32'
pub async fn bar(a: i32, b: i32) -> i32 {
    0
}

// @has async_fn/fn.baz.html '//pre[@class="rust fn"]' 'pub async fn baz<T>(a: T) -> T'
pub async fn baz<T>(a: T) -> T {
    a
}

trait Bar {}

impl Bar for () {}

// @has async_fn/fn.quux.html '//pre[@class="rust fn"]' 'pub async fn quux() -> impl Bar'
pub async fn quux() -> impl Bar {
    ()
}

// @has async_fn/struct.Foo.html
// @matches - '//code' 'pub async fn f\(\)$'
pub struct Foo;

impl Foo {
    pub async fn f() {}
}
