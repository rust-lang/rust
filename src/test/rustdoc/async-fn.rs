// edition:2018
#![feature(min_const_generics)]

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

// @has async_fn/fn.qux.html '//pre[@class="rust fn"]' 'pub async unsafe fn qux() -> char'
pub async unsafe fn qux() -> char {
    '⚠'
}

// @has async_fn/fn.mut_args.html '//pre[@class="rust fn"]' 'pub async fn mut_args(a: usize)'
pub async fn mut_args(mut a: usize) {}

// @has async_fn/fn.mut_ref.html '//pre[@class="rust fn"]' 'pub async fn mut_ref(x: i32)'
pub async fn mut_ref(ref mut x: i32) {}

trait Bar {}

impl Bar for () {}

// @has async_fn/fn.quux.html '//pre[@class="rust fn"]' 'pub async fn quux() -> impl Bar'
pub async fn quux() -> impl Bar {
    ()
}

// @has async_fn/struct.Foo.html
// @matches - '//code' 'pub async fn f\(\)$'
// @matches - '//code' 'pub async unsafe fn g\(\)$'
// @matches - '//code' 'pub async fn mut_self\(self, first: usize\)$'
pub struct Foo;

impl Foo {
    pub async fn f() {}
    pub async unsafe fn g() {}
    pub async fn mut_self(mut self, mut first: usize) {}
}

pub trait Trait<const N: usize> {}
// @has async_fn/fn.const_generics.html
// @has - '//pre[@class="rust fn"]' 'pub async fn const_generics<const N: usize>(_: impl Trait<N>)'
pub async fn const_generics<const N: usize>(_: impl Trait<N>) {}
