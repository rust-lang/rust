//check-pass

#![feature(type_alias_impl_trait)]

trait Trait {
    type Opaque1;
    type Opaque2;
    fn constrain(self);
}

impl<'a> Trait for &'a () {
    type Opaque1 = impl Sized;
    type Opaque2 = impl Sized + 'a;
    fn constrain(self) {
        let _: Self::Opaque1 = ();
        let _: Self::Opaque2 = self;
    }
}

fn main() {}
