pub trait T {}

struct S<'a>(&'a ());

impl<'a> T for S<'a> {}

fn foo() -> impl T {
    let x = ();
    S(&x) //~ ERROR `x` does not live long enough
}

fn main() {}
