//! Since there is only one possible `bar` method, we invoke it and subsequently
//! constrain `foo`'s RPIT to `u32`.

//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ check-pass

trait Trait {}

impl Trait for u32 {}

struct Bar<T>(T);

impl Bar<u32> {
    fn bar(self) {}
}

fn foo(x: bool) -> Bar<impl Sized> {
    if x {
        let x = foo(false);
        x.bar();
    }
    todo!()
}

fn main() {}
