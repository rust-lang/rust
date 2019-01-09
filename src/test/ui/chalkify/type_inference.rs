// compile-flags: -Z chalk

trait Foo { }
impl Foo for i32 { }

trait Bar { }
impl Bar for i32 { }
impl Bar for u32 { }

fn only_foo<T: Foo>(_x: T) { }

fn only_bar<T: Bar>(_x: T) { }

fn main() {
    let x = 5.0;

    // The only type which implements `Foo` is `i32`, so the chalk trait solver
    // is expecting a variable of type `i32`. This behavior differs from the
    // old-style trait solver. I guess this will change, that's why I'm
    // adding that test.
    only_foo(x); //~ ERROR mismatched types

    // Here we have two solutions so we get back the behavior of the old-style
    // trait solver.
    only_bar(x); //~ ERROR the trait bound `{float}: Bar` is not satisfied
}
