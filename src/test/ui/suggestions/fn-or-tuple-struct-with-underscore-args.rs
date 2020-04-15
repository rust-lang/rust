fn foo(a: usize, b: usize) -> usize { a }

struct S(usize, usize);

trait T {
    fn baz(x: usize, y: usize) -> usize { x }
}

fn main() {
    let _: usize = foo(_, _);
    //~^ ERROR expected expression
    //~| ERROR expected expression
    //~| ERROR destructuring assignments are unstable
    //~| ERROR destructuring assignments are unstable
    let _: S = S(_, _);
    //~^ ERROR expected expression
    //~| ERROR expected expression
    //~| ERROR destructuring assignments are unstable
    //~| ERROR destructuring assignments are unstable
    let _: usize = T::baz(_, _);
    //~^ ERROR expected expression
    //~| ERROR expected expression
    //~| ERROR destructuring assignments are unstable
    //~| ERROR destructuring assignments are unstable
}
