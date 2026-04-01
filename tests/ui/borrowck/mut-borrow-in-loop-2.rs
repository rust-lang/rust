#![allow(dead_code)]

struct Events<R>(R);

struct Other;

pub trait Trait<T> {
    fn handle(value: T) -> Self;
}

// Blanket impl. (If you comment this out, compiler figures out that it
// is passing an `&mut` to a method that must be expecting an `&mut`,
// and injects an auto-reborrow.)
impl<T, U> Trait<U> for T where T: From<U> {
    fn handle(_: U) -> Self { unimplemented!() }
}

impl<'a, R> Trait<&'a mut Events<R>> for Other {
    fn handle(_: &'a mut Events<R>) -> Self { unimplemented!() }
}

fn this_compiles<'a, R>(value: &'a mut Events<R>) {
    for _ in 0..3 {
        Other::handle(&mut *value);
    }
}

fn this_does_not<'a, R>(value: &'a mut Events<R>) {
    for _ in 0..3 {
        Other::handle(value); //~ ERROR use of moved value: `value`
    }
}

fn main() {}
