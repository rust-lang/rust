// Regression test for #129541
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

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
    //~^ ERROR cycle detected when computing layout of `Hello` [E0391]
    Variant(<[Hello] as Normalize>::Assoc),
}

fn main() {}
