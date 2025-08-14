//! A simple test for testing many permutations of allowedness of
//! impl Trait
#![feature(impl_trait_in_fn_trait_return)]
#![feature(custom_inner_attributes)]
#![rustfmt::skip]
use std::fmt::Debug;

// Allowed
fn in_parameters(_: impl Debug) { panic!() }

// Allowed
fn in_return() -> impl Debug { panic!() }

// Allowed
fn in_adt_in_parameters(_: Vec<impl Debug>) { panic!() }

// Disallowed
fn in_fn_parameter_in_parameters(_: fn(impl Debug)) { panic!() }
//~^ ERROR `impl Trait` is not allowed in `fn` pointer

// Disallowed
fn in_fn_return_in_parameters(_: fn() -> impl Debug) { panic!() }
//~^ ERROR `impl Trait` is not allowed in `fn` pointer

// Disallowed
fn in_fn_parameter_in_return() -> fn(impl Debug) { panic!() }
//~^ ERROR `impl Trait` is not allowed in `fn` pointer

// Disallowed
fn in_fn_return_in_return() -> fn() -> impl Debug { panic!() }
//~^ ERROR `impl Trait` is not allowed in `fn` pointer

// Disallowed
fn in_dyn_Fn_parameter_in_parameters(_: &dyn Fn(impl Debug)) { panic!() }
//~^ ERROR `impl Trait` is not allowed in the parameters of `Fn` trait bounds

// Disallowed
fn in_dyn_Fn_return_in_parameters(_: &dyn Fn() -> impl Debug) { panic!() }
//~^ ERROR `impl Trait` is not allowed in the return type of `Fn` trait bounds

// Disallowed
fn in_dyn_Fn_parameter_in_return() -> &'static dyn Fn(impl Debug) { panic!() }
//~^ ERROR `impl Trait` is not allowed in the parameters of `Fn` trait bounds

// Allowed (but it's still ambiguous; nothing constrains the RPIT in this body).
fn in_dyn_Fn_return_in_return() -> &'static dyn Fn() -> impl Debug { panic!() }
//~^ ERROR: type annotations needed

// Disallowed
fn in_impl_Fn_parameter_in_parameters(_: &impl Fn(impl Debug)) { panic!() }
//~^ ERROR `impl Trait` is not allowed in the parameters of `Fn` trait bounds
//~^^ ERROR nested `impl Trait` is not allowed

// Disallowed
fn in_impl_Fn_return_in_parameters(_: &impl Fn() -> impl Debug) { panic!() }
//~^ ERROR `impl Trait` is not allowed in the return type of `Fn` trait bounds

// Disallowed
fn in_impl_Fn_parameter_in_return() -> &'static impl Fn(impl Debug) { panic!() }
//~^ ERROR `impl Trait` is not allowed in the parameters of `Fn` trait bounds
//~| ERROR nested `impl Trait` is not allowed

// Allowed
fn in_impl_Fn_return_in_return() -> &'static impl Fn() -> impl Debug { panic!() }
//~^ ERROR: type annotations needed

// Disallowed
fn in_Fn_parameter_in_generics<F: Fn(impl Debug)> (_: F) { panic!() }
//~^ ERROR `impl Trait` is not allowed in the parameters of `Fn` trait bounds

// Disallowed
fn in_Fn_return_in_generics<F: Fn() -> impl Debug> (_: F) { panic!() }
//~^ ERROR `impl Trait` is not allowed in the return type of `Fn` trait bounds


// Allowed
fn in_impl_Trait_in_parameters(_: impl Iterator<Item = impl Iterator>) { panic!() }

// Allowed
fn in_impl_Trait_in_return() -> impl IntoIterator<Item = impl IntoIterator> {
    vec![vec![0; 10], vec![12; 7], vec![8; 3]]
}

// Disallowed
struct InBraceStructField { x: impl Debug }
//~^ ERROR `impl Trait` is not allowed in field types

// Disallowed
struct InAdtInBraceStructField { x: Vec<impl Debug> }
//~^ ERROR `impl Trait` is not allowed in field types

// Disallowed
struct InTupleStructField(impl Debug);
//~^ ERROR `impl Trait` is not allowed in field types

// Disallowed
enum InEnum {
    InBraceVariant { x: impl Debug },
    //~^ ERROR `impl Trait` is not allowed in field types
    InTupleVariant(impl Debug),
    //~^ ERROR `impl Trait` is not allowed in field types
}

// Allowed
trait InTraitDefnParameters {
    fn in_parameters(_: impl Debug);
}

// Allowed
trait InTraitDefnReturn {
    fn in_return() -> impl Debug;
}

// Allowed and disallowed in trait impls
trait DummyTrait {
    type Out;
    fn in_trait_impl_parameter(_: impl Debug);
    fn in_trait_impl_return() -> Self::Out;
}
impl DummyTrait for () {
    type Out = impl Debug;
    //~^ ERROR `impl Trait` in associated types is unstable
    //~| ERROR unconstrained opaque type

    fn in_trait_impl_parameter(_: impl Debug) { }
    // Allowed

    fn in_trait_impl_return() -> impl Debug { () }
    //~^ ERROR `in_trait_impl_return` has an incompatible type for trait
    // Allowed
}

// Allowed
struct DummyType;
impl DummyType {
    fn in_inherent_impl_parameters(_: impl Debug) { }
    fn in_inherent_impl_return() -> impl Debug { () }
}

// Disallowed
extern "C" {
    fn in_foreign_parameters(_: impl Debug);
    //~^ ERROR `impl Trait` is not allowed in `extern fn`

    fn in_foreign_return() -> impl Debug;
    //~^ ERROR `impl Trait` is not allowed in `extern fn`
}

// Allowed
extern "C" fn in_extern_fn_parameters(_: impl Debug) {
}

// Allowed
extern "C" fn in_extern_fn_return() -> impl Debug {
    22
}

type InTypeAlias<R> = impl Debug;
//~^ ERROR `impl Trait` in type aliases is unstable
//~| ERROR unconstrained opaque type

type InReturnInTypeAlias<R> = fn() -> impl Debug;
//~^ ERROR `impl Trait` is not allowed in `fn` pointer
//~| ERROR `impl Trait` in type aliases is unstable

// Disallowed in impl headers
impl PartialEq<impl Debug> for () {
    //~^ ERROR `impl Trait` is not allowed in traits
}

// Disallowed in impl headers
impl PartialEq<()> for impl Debug {
    //~^ ERROR `impl Trait` is not allowed in impl headers
}

// Disallowed in inherent impls
impl impl Debug {
    //~^ ERROR `impl Trait` is not allowed in impl headers
}

// Disallowed in inherent impls
struct InInherentImplAdt<T> { t: T }
impl InInherentImplAdt<impl Debug> {
    //~^ ERROR `impl Trait` is not allowed in impl headers
}

// Disallowed in where clauses
fn in_fn_where_clause()
    where impl Debug: Debug
//~^ ERROR `impl Trait` is not allowed in bounds
{
}

// Disallowed in where clauses
fn in_adt_in_fn_where_clause()
    where Vec<impl Debug>: Debug
//~^ ERROR `impl Trait` is not allowed in bounds
{
}

// Disallowed
fn in_trait_parameter_in_fn_where_clause<T>()
    where T: PartialEq<impl Debug>
//~^ ERROR `impl Trait` is not allowed in bounds
{
}

// Disallowed
fn in_Fn_parameter_in_fn_where_clause<T>()
    where T: Fn(impl Debug)
//~^ ERROR `impl Trait` is not allowed in the parameters of `Fn` trait bounds
{
}

// Disallowed
fn in_Fn_return_in_fn_where_clause<T>()
    where T: Fn() -> impl Debug
//~^ ERROR `impl Trait` is not allowed in the return type of `Fn` trait bounds
{
}

// Disallowed
struct InStructGenericParamDefault<T = impl Debug>(T);
//~^ ERROR `impl Trait` is not allowed in generic parameter defaults

// Disallowed
enum InEnumGenericParamDefault<T = impl Debug> { Variant(T) }
//~^ ERROR `impl Trait` is not allowed in generic parameter defaults

// Disallowed
trait InTraitGenericParamDefault<T = impl Debug> {}
//~^ ERROR `impl Trait` is not allowed in generic parameter defaults

// Disallowed
type InTypeAliasGenericParamDefault<T = impl Debug> = T;
//~^ ERROR `impl Trait` is not allowed in generic parameter defaults

// Disallowed
#[expect(invalid_type_param_default)]
impl<T = impl Debug> T {}
//~^ ERROR `impl Trait` is not allowed in generic parameter defaults
//~| ERROR no nominal type found

// Disallowed
#[expect(invalid_type_param_default)]
fn in_method_generic_param_default<T = impl Debug>(_: T) {}
//~^ ERROR `impl Trait` is not allowed in generic parameter defaults

fn main() {
    let _in_local_variable: impl Fn() = || {};
    //~^ ERROR `impl Trait` is not allowed in the type of variable bindings
    let _in_return_in_local_variable = || -> impl Fn() { || {} };
    //~^ ERROR `impl Trait` is not allowed in closure return types
}
