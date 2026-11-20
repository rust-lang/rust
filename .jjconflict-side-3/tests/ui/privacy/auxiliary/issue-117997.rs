//@ no-prefer-dynamic
//@ compile-flags: --crate-type=rlib

pub use impl_mod::TraitImplementer as Implementer;

pub use trait_mod::get_assoc;

mod impl_mod {
    use crate::trait_mod::TraitWithAssocType;

    pub struct TraitImplementer {}
    pub struct AssociatedType {}

    impl AssociatedType {
        pub fn method_on_assoc(&self) -> i32 {
            todo!()
        }
    }

    impl TraitWithAssocType for TraitImplementer {
        type AssocType = AssociatedType;
    }
}

mod trait_mod {
    use crate::Implementer;

    pub fn get_assoc() -> <Implementer as TraitWithAssocType>::AssocType {
        todo!()
    }

    pub trait TraitWithAssocType {
        type AssocType;
    }
}
