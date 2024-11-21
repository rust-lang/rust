trait Trait {
    type Assoc;
    fn generate(&self) -> Self::Assoc;
}

trait Other {}

impl<S> Trait for S where S: Other + ?Sized {
    type Assoc = &'static str;
    fn generate(&self) -> Self::Assoc { "hi" }
}

trait Downstream: Trait<Assoc = usize> {}
impl<T> Other for T where T: ?Sized + Downstream + OnlyDyn {}

trait OnlyDyn {}
impl OnlyDyn for dyn Downstream {}

struct Concrete;
impl Trait for Concrete {
    type Assoc = usize;
    fn generate(&self) -> Self::Assoc { 42 }
}
impl Downstream for Concrete {}

fn test<T: ?Sized + Other>(x: &T) {
    let s: &str = x.generate();
    println!("{s}");
}

fn impl_downstream<T: ?Sized + Downstream>(x: &T) {}

fn main() {
    let x: &dyn Downstream = &Concrete;

    test(x); // This call used to segfault.
    //~^ ERROR type mismatch resolving

    // This no longer holds since `Downstream: Trait<Assoc = usize>`,
    // but the `Trait<Assoc = &'static str>` blanket impl now shadows.
    impl_downstream(x);
    //~^ ERROR type mismatch resolving
}
