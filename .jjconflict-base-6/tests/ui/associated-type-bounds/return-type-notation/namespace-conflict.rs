//@ check-pass

#![allow(non_camel_case_types)]
#![feature(return_type_notation)]

trait Foo {
    type test;

    fn test() -> impl Bar;
}

fn call_path<T: Foo>()
where
    T::test(..): Bar,
{
}

fn call_bound<T: Foo<test(..): Bar>>() {}

trait Bar {}
struct NotBar;
struct YesBar;
impl Bar for YesBar {}

impl Foo for () {
    type test = NotBar;

    // Use refinement here so we can observe `YesBar: Bar`.
    #[allow(refining_impl_trait_internal)]
    fn test() -> YesBar {
        YesBar
    }
}

fn main() {
    // If `T::test(..)` resolved to the GAT (erroneously), then this would be
    // an error since `<() as Foo>::bar` -- the associated type -- does not
    // implement `Bar`, but the return type of the method does.
    call_path::<()>();
    call_bound::<()>();
}
