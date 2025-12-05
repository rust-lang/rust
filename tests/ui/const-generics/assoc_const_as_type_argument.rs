trait Trait {
    const ASSOC: usize;
}

fn bar<const N: usize>() {}

fn foo<T: Trait>() {
    bar::<<T as Trait>::ASSOC>();
    //~^ ERROR: expected associated type, found associated constant `Trait::ASSOC`
    //~| ERROR: unresolved item provided when a constant was expected
}

fn main() {}
