//@ check-pass

trait Trait {
    type Assoc;
}

impl Trait for () {
    type Assoc = ();
}

fn unit() -> impl Into<<() as Trait>::Assoc> {}

pub fn ice() {
    Into::into(unit());
}

fn main() {}
