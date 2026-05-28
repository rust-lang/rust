#![feature(min_generic_const_args, adt_const_params)]
#![expect(incomplete_features)]
#![crate_type = "lib"]

// Miscellaneous assortment of invalid cases of directly represented
// `ConstArgKind::Struct`'s under mgca.

#[derive(Eq, PartialEq, std::marker::ConstParamTy)]
struct Foo<T> { field: T }

fn NonStruct() {}

fn accepts<const N: Foo<u8>>() {}

fn bar() {
    accepts::<{ Foo::<u8> { }}>();
    //~^ ERROR: struct expression with missing field initialiser for `field`
    accepts::<{ Foo::<u8> { field: const { 1 }, field: const { 2} }}>();
    //~^ ERROR: struct expression with multiple initialisers for `field`
    accepts::<{ Fooo::<u8> { field: const { 1 } }}>();
    //~^ ERROR: cannot find struct, variant or union type `Fooo` in this scope
    //~| ERROR: struct expression with invalid base path
    accepts::<{ NonStruct { }}>();
    //~^ ERROR: expected struct, variant or union type, found function `NonStruct`
    //~| ERROR: struct expression with invalid base path
}
