// Regression test for #129541

//@ check-pass

trait Bound {}
trait Normalize {
    type Assoc;
}

impl<T: Bound> Normalize for T {
    type Assoc = T;
}

impl<T: Bound> Normalize for [T] {
    type Assoc = T;
}

impl Bound for Hello {}
struct Hello {
    a: <[Hello] as Normalize>::Assoc,
}

fn main() {}
