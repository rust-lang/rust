// check-pass
// known-bug: #101350

trait Trait {
    type Ty;
}

impl Trait for &'static () {
    type Ty = ();
}

fn extend<'a>() {
    None::<<&'a () as Trait>::Ty>;
}

fn main() {}
