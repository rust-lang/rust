fn foo(a: usize, b: usize) -> usize { a }

struct S(usize, usize);

trait T {
    fn baz(x: usize, y: usize) -> usize { x }
}

fn main() {
    let _: usize = foo(_, _);
    //~^ ERROR `_` can only be used on the left-hand side of an assignment
    //~| ERROR `_` can only be used on the left-hand side of an assignment
    let _: S = S(_, _);
    //~^ ERROR `_` can only be used on the left-hand side of an assignment
    //~| ERROR `_` can only be used on the left-hand side of an assignment
    let _: usize = T::baz(_, _);
    //~^ ERROR `_` can only be used on the left-hand side of an assignment
    //~| ERROR `_` can only be used on the left-hand side of an assignment
}
