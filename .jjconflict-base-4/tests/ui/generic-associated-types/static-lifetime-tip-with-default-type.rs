//@ error-pattern: r#T: 'static
//@ error-pattern: r#K: 'static
//@ error-pattern: T: 'static=

// https://github.com/rust-lang/rust/issues/124785

struct Foo<T, K = i32>(&'static T, &'static K);
//~^ ERROR: the parameter type `T` may not live long enough
//~| ERROR: the parameter type `K` may not live long enough

struct Bar<r#T, r#K = i32>(&'static r#T, &'static r#K);
//~^ ERROR: the parameter type `T` may not live long enough
//~| ERROR: the parameter type `K` may not live long enough

struct Boo<T= i32>(&'static T);
//~^ ERROR: the parameter type `T` may not live long enough

struct Far<T
= i32>(&'static T);
//~^ ERROR: the parameter type `T` may not live long enough

struct S<'a, K: 'a = i32>(&'static K);
//~^ ERROR: lifetime parameter `'a` is never used
//~| ERROR: the parameter type `K` may not live long enough

fn main() {}
