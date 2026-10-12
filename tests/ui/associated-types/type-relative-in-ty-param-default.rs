// Test that we can resolve type-relative associated type paths inside type parameter defaults
// where the self type is a simple type parameter.
//
// issue: <https://github.com/rust-lang/rust/issues/87682>

//@ check-pass

trait Trait {
    type Type;
}

// Below, `T::Type` resolves to `<T as Trait>::Type` since the owner has bound `T: Trait`.

struct Owner0<T: Trait, U = T::Type>(T, U);

struct Owner1<T, U = T::Type>(T, U)
where
    T: Trait;

fn main() {}
