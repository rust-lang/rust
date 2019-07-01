// run-pass

// Check that resolving, in the value namespace, to an `enum` variant
// through a type alias is well behaved in the presence of generics.
// We check for situations with:
// 1. a generic type `Alias<T>`, we can type-apply `Alias` when referring to a variant.
// 2. a monotype `AliasFixed` of generic `Enum<T>`, we can refer to variants
//    and the type-application of `T` in `AliasFixed` is kept.

#![allow(irrefutable_let_patterns)]

enum Enum<T> { TSVariant(T), SVariant { v: T }, UVariant }
type Alias<T> = Enum<T>;
type AliasFixed = Enum<()>;

macro_rules! is_variant {
    (TSVariant, $expr:expr) => (is_variant!(@check TSVariant, (_), $expr));
    (SVariant, $expr:expr) => (is_variant!(@check SVariant, { v: _ }, $expr));
    (UVariant, $expr:expr) => (is_variant!(@check UVariant, {}, $expr));
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

    // Unit variant

    is_variant!(UVariant, Enum::UVariant);
    is_variant!(UVariant, Enum::UVariant::<()>);
    is_variant!(UVariant, Enum::<()>::UVariant);

    is_variant!(UVariant, Alias::UVariant);
    is_variant!(UVariant, Alias::<()>::UVariant);

    is_variant!(UVariant, AliasFixed::UVariant);
}
