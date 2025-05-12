//@check-pass

#![feature(impl_trait_in_assoc_type)]

trait Trait {
    type Opaque1;
    type Opaque2;
    fn constrain(self) -> (Self::Opaque1, Self::Opaque2);
}

impl<'a> Trait for &'a () {
    type Opaque1 = impl Sized;
    type Opaque2 = impl Sized + 'a;
    fn constrain(self) -> (Self::Opaque1, Self::Opaque2) {
        let a: Self::Opaque1 = ();
        let b: Self::Opaque2 = self;
        (a, b)
    }
}

fn main() {}
