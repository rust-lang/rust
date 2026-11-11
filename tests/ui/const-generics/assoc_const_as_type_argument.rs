trait Trait {
    const ASSOC: usize;
}

fn bar<const N: usize>() {}

fn foo<T: Trait>() {
    bar::<<T as Trait>::ASSOC>();
    //~^ ERROR: cannot find associated type `ASSOC` in trait `Trait`
}

fn main() {}
