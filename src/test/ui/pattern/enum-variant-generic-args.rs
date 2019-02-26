// run-pass

#![feature(type_alias_enum_variants)]

#![allow(irrefutable_let_patterns)]

#[allow(dead_code)]
enum Enum<T> { TSVariant(T), SVariant { v: T } }
type Alias<T> = Enum<T>;
type AliasFixed = Enum<()>;

macro_rules! is_variant {
    (TSVariant, $expr:expr) => (is_variant!(@check TSVariant, (_), $expr));
    (SVariant, $expr:expr) => (is_variant!(@check SVariant, { v: _ }, $expr));
    (@check $variant:ident, $matcher:tt, $expr:expr) => (
        assert!(if let Enum::$variant::<()> $matcher = $expr { true } else { false },
                "expr does not have correct type");
    );
}

fn main() {
    // Tuple struct variant

    is_variant!(TSVariant, Enum::TSVariant(()));
    is_variant!(TSVariant, Enum::TSVariant::<()>(()));
    is_variant!(TSVariant, Enum::<()>::TSVariant(()));

    is_variant!(TSVariant, Alias::TSVariant(()));
    is_variant!(TSVariant, Alias::<()>::TSVariant(()));

    is_variant!(TSVariant, AliasFixed::TSVariant(()));

    // Struct variant

    is_variant!(SVariant, Enum::SVariant { v: () });
    is_variant!(SVariant, Enum::SVariant::<()> { v: () });
    is_variant!(SVariant, Enum::<()>::SVariant { v: () });

    is_variant!(SVariant, Alias::SVariant { v: () });
    is_variant!(SVariant, Alias::<()>::SVariant { v: () });

    is_variant!(SVariant, AliasFixed::SVariant { v: () });
}
