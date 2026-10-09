// Test that we can resolve type-relative associated type paths inside const parameter types
// where the self type is a simple type parameter.

//@ check-pass
#![feature(generic_const_parameter_types, adt_const_params, const_param_ty_trait)]

trait Trait {
    type Type;
}

// Below, `T::Type` resolves to `<T as Trait>::Type` since the owner has bound `T: Trait`.

struct Owner0<T, const N: T::Type>(T)
where
    T: Trait<Type = usize>;

struct Owner1<T, const N: T::Type>(T)
where
    T: Trait<Type: std::marker::ConstParamTy_>;

fn main() {}
