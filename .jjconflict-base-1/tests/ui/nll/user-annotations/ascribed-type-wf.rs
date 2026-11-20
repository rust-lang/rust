// Regression test for #101350.
//@ check-fail

trait Trait {
    type Ty;
}

impl Trait for &'static () {
    type Ty = ();
}

fn extend<'a>() {
    None::<<&'a () as Trait>::Ty>;
    //~^ ERROR lifetime may not live long enough
}

fn main() {}
