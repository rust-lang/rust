#![feature(irrefutable_let_patterns)]
#![feature(type_alias_enum_variants)]

#![allow(irrefutable_let_patterns)]

enum Enum<T> { Variant(T) }
type Alias<T> = Enum<T>;
type AliasFixed = Enum<()>;

macro_rules! is_variant {
    ($expr:expr) => (
        assert!(if let Enum::Variant::<()>(_) = $expr { true } else { false },
                "expr does not have correct type");
    )
}

impl<T> Enum<T> {
    fn foo() {
        is_variant!(Self::Variant(()));
    }
}

fn main() {
    is_variant!(Enum::Variant(()));
    is_variant!(Enum::Variant::<()>(()));
    is_variant!(Enum::<()>::Variant(()));

    is_variant!(Alias::Variant(()));
    is_variant!(Alias::<()>::Variant(()));

    is_variant!(AliasFixed::Variant(()));

    Enum::<()>::foo();
}
