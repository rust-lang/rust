// Regression test for #129541

//@ revisions: unique multiple
//@ error-pattern: reached the recursion limit finding the struct tail for `<[Hello] as Normalize>::Assoc`

trait Bound {}
trait Normalize {
    type Assoc;
}

#[cfg(multiple)]
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
