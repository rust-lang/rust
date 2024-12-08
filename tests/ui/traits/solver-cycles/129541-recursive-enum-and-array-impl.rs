// Regression test for #129541
//~^ ERROR cycle detected when computing layout of `<[Hello] as Normalize>::Assoc` [E0391]

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
enum Hello {
    Variant(<[Hello] as Normalize>::Assoc),
}

fn main() {}
