trait Trait {
    const ASSOC: usize;
}

fn bar<const N: usize>() {}

fn foo<T: Trait>() {
    bar::<<T as Trait>::ASSOC>();
    //~^ ERROR: expected associated type, found associated constant `Trait::ASSOC`
}

fn main() {}
