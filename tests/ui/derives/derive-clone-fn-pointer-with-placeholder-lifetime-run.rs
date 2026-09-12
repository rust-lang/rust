//@ run-pass
// Cloned function pointer fields works fine even when the function's
// input type is not Clone.

#![allow(dead_code)]

trait SomeTrait {
    type SomeType<'a>;
}

#[derive(Clone)]
struct Concrete;

struct NotClone<'a> {
    value: &'a u32,
}

impl SomeTrait for Concrete {
    type SomeType<'a> = NotClone<'a>;
}

fn read_value(x: NotClone<'_>) -> u32 {
    *x.value
}

#[derive(Clone)]
struct Foo<T: SomeTrait> {
    x: fn(T::SomeType<'_>) -> u32,
    explicit: for<'a> fn(T::SomeType<'a>) -> u32,
}

fn main() {
    let foo = Foo::<Concrete> { x: read_value, explicit: read_value };
    let cloned = foo.clone();

    let n = 42;
    assert_eq!((cloned.x)(NotClone { value: &n }), 42);
    assert_eq!((cloned.explicit)(NotClone { value: &n }), 42);
}
