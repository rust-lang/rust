//@ edition:2018
trait T {
    type O;
}

struct S;

impl T for S {
    type O = ();
}

fn foo() -> impl T<O=()> { S }

fn bar(f: impl T<O=()>) {}

fn main() {
    bar(foo); //~ERROR E0277
    let closure = || S;
    bar(closure); //~ERROR E0277
}
