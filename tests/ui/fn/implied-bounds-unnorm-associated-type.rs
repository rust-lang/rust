//@ check-fail
// See issue #91068. We check that the unnormalized associated types in
// function signatures are implied

trait Trait {
    type Type;
}

impl<T> Trait for T {
    type Type = ();
}

fn f<'a, 'b>(s: &'b str, _: <&'a &'b () as Trait>::Type) -> &'a str {
    s
}

fn main() {
    let x = String::from("Hello World!");
    let y = f(&x, ());
    drop(x);
    //~^ ERROR cannot move out of `x` because it is borrowed
    println!("{}", y);
}
