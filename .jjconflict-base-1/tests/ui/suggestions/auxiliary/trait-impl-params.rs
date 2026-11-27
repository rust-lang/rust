//@ edition:2018
pub trait Foo {
    fn foo(a: i32, b: u32);
}

pub trait Bar {
    fn no_params();
    fn has_self(&self);
    fn has_two_self(self: &Self, other: &Self);
}

pub trait Baz<T> {
    fn generic_no_params<U: Baz<()>>();
    fn generic_one_param<U: Baz<()>>(a: U);
}

pub trait NonIdentArguments {
    fn pattern_types_in_arguments(_: i32, (_a, _b): (i32, i32)) {}
}
