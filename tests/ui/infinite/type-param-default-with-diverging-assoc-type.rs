//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)

trait Trait {
    type Diverges<D: Trait>;
}

impl<T> Trait for T {
    type Diverges<D: Trait> = D::Diverges<D>;
}

struct Bar<T: ?Sized = <u8 as Trait>::Diverges<u8>>(Box<T>);
//[current]~^ ERROR overflow evaluating the requirement
//[next]~^^ ERROR type mismatch resolving

fn main() {}
