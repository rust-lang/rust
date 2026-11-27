// Ensure that we detect and reject implicit forward references to generic parameters in const
// parameter types arising from type-relative associated item paths.

#![feature(generic_const_parameter_types, adt_const_params, const_param_ty_trait)]

// Below, `T::Type` resolves to `<T as Trait<N>>::Type` but is a forward reference.
struct Def0<T: Trait<N>, const N: T::Type>(T);
//~^ ERROR const parameter type references generic parameter `N` before it is declared
struct Use0(Def0<(), 0>);

trait Trait<const N: usize> { type Type; }
impl Trait<0> for () { type Type = u32; }

fn main() {}
