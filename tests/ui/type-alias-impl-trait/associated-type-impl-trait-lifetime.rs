//check-pass

#![feature(type_alias_impl_trait)]

trait Trait {
    type Opaque1;
    type Opaque2;
    fn constrain(self) -> (Self::Opaque1, Self::Opaque2);
}

impl<'a> Trait for &'a () {
    type Opaque1 = impl Sized;
    type Opaque2 = impl Sized + 'a;
    fn constrain(self) -> (Self::Opaque1, Self::Opaque2) {
        ((), self)
    }
}

fn main() {}
