// Normally we do not allow equal lifetimes in opaque type generic args at
// their defining sites. An exception to this rule, however, is when the bounds
// of the opaque type *require* the lifetimes to be equal.
// issue: #113916
//@ check-pass

#![feature(type_alias_impl_trait)]
#![feature(impl_trait_in_assoc_type)]

trait Trait<'a, 'b> {}
impl<T> Trait<'_, '_> for T {}

mod equal_params {
    type Opaque<'a: 'b, 'b: 'a> = impl super::Trait<'a, 'b>;
    #[define_opaque(Opaque)]
    fn test<'a: 'b, 'b: 'a>() -> Opaque<'a, 'b> {
        let _ = None::<&'a &'b &'a ()>;
        0u8
    }
}

mod equal_static {
    type Opaque<'a: 'static> = impl Sized + 'a;
    #[define_opaque(Opaque)]
    fn test<'a: 'static>() -> Opaque<'a> {
        let _ = None::<&'static &'a ()>;
        0u8
    }
}

mod implied_bounds {
    trait Traitor {
        type Assoc;
        fn define(self) -> Self::Assoc;
    }

    impl<'a> Traitor for &'static &'a () {
        type Assoc = impl Sized + 'a;
        fn define(self) -> Self::Assoc {
            let _ = None::<&'static &'a ()>;
            0u8
        }
    }

    impl<'a, 'b> Traitor for (&'a &'b (), &'b &'a ()) {
        type Assoc = impl Sized + 'a + 'b;
        fn define(self) -> Self::Assoc {
            let _ = None::<(&'a &'b (), &'b &'a ())>;
            0u8
        }
    }
}

fn main() {}
