// build-pass (FIXME(62277): could be check-pass?)
// edition:2018
// compile-flags: --crate-type lib

#![feature(in_band_lifetimes)]

use std::future::Future;

pub async fn simple_generic<T>() {}

pub trait Foo {
    fn foo(&self) {}
}

struct FooType;
impl Foo for FooType {}

pub async fn call_generic_bound<F: Foo>(f: F) {
    f.foo()
}

pub async fn call_where_clause<F>(f: F)
where
    F: Foo,
{
    f.foo()
}

pub async fn call_impl_trait(f: impl Foo) {
    f.foo()
}

pub async fn call_with_ref(f: &impl Foo) {
    f.foo()
}

pub fn async_fn_with_same_generic_params_unifies() {
    let mut a = call_generic_bound(FooType);
    a = call_generic_bound(FooType);

    let mut b = call_where_clause(FooType);
    b = call_where_clause(FooType);

    let mut c = call_impl_trait(FooType);
    c = call_impl_trait(FooType);

    let f_one = FooType;
    let f_two = FooType;
    let mut d = call_with_ref(&f_one);
    d = call_with_ref(&f_two);
}

pub fn simple_generic_block<T>() -> impl Future<Output = ()> {
    async move {}
}

pub fn call_generic_bound_block<F: Foo>(f: F) -> impl Future<Output = ()> {
    async move { f.foo() }
}

pub fn call_where_clause_block<F>(f: F) -> impl Future<Output = ()>
where
    F: Foo,
{
    async move { f.foo() }
}

pub fn call_impl_trait_block(f: impl Foo) -> impl Future<Output = ()> {
    async move { f.foo() }
}

pub fn call_with_ref_block<'a>(f: &'a (impl Foo + 'a)) -> impl Future<Output = ()> + 'a {
    async move { f.foo() }
}

pub fn call_with_ref_block_in_band(f: &'a (impl Foo + 'a)) -> impl Future<Output = ()> + 'a {
    async move { f.foo() }
}

pub fn async_block_with_same_generic_params_unifies() {
    let mut a = call_generic_bound_block(FooType);
    a = call_generic_bound_block(FooType);

    let mut b = call_where_clause_block(FooType);
    b = call_where_clause_block(FooType);

    let mut c = call_impl_trait_block(FooType);
    c = call_impl_trait_block(FooType);

    let f_one = FooType;
    let f_two = FooType;
    let mut d = call_with_ref_block(&f_one);
    d = call_with_ref_block(&f_two);

    let f_one = FooType;
    let f_two = FooType;
    let mut d = call_with_ref_block_in_band(&f_one);
    d = call_with_ref_block_in_band(&f_two);
}
