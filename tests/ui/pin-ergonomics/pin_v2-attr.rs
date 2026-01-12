#![feature(
    pin_ergonomics,
    where_clause_attrs,
    trait_alias,
    extern_types,
    associated_type_defaults,
    fn_delegation,
)]
#![allow(incomplete_features)]
#![pin_v2] //~ ERROR `#[pin_v2]` attribute cannot be used on crates

// allowed

#[pin_v2]
struct Struct {}

#[pin_v2]
enum Enum {}

#[pin_v2]
union Union {
    field: (),
}

// disallowed

enum Foo<#[pin_v2] T, #[pin_v2] U = ()> {
    //~^ ERROR `#[pin_v2]` attribute cannot be used on type parameters
    //~| ERROR `#[pin_v2]` attribute cannot be used on type parameters
    #[pin_v2] //~ ERROR `#[pin_v2]` attribute cannot be used on enum variants
    UnitVariant,
    TupleVariant(#[pin_v2] T), //~ ERROR `#[pin_v2]` attribute cannot be used on struct fields
    StructVariant {
        #[pin_v2] //~ ERROR `#[pin_v2]` attribute cannot be used on struct fields
        field: U,
    },
}

#[pin_v2] //~ ERROR `#[pin_v2]` attribute cannot be used on traits
trait Trait {
    #[pin_v2] //~ ERROR `#[pin_v2]` attribute cannot be used on associated consts
    const ASSOC_CONST: () = ();
    #[pin_v2] //~ ERROR `#[pin_v2]` attribute cannot be used on associated types
    type AssocType = ();

    #[pin_v2] //~ ERROR `#[pin_v2]` attribute cannot be used on required trait methods
    fn method();
    #[pin_v2] //~ ERROR `#[pin_v2]` attribute cannot be used on provided trait methods
    fn method_with_body() {}
}

#[pin_v2] //~ ERROR `#[pin_v2]` attribute cannot be used on trait aliases
trait TraitAlias = Trait;

#[pin_v2] //~ ERROR `#[pin_v2]` attribute cannot be used on inherent impl blocks
impl Struct {
    // FIXME: delegation macros are not tested yet (how to?)
    #[pin_v2] //~ ERROR `#[pin_v2]` attribute cannot be used on delegations
    reuse <Struct as std::any::Any>::type_id;

    #[pin_v2] //~ ERROR `#[pin_v2]` attribute cannot be used on inherent methods
    fn method() {}
}

#[pin_v2] //~ ERROR `#[pin_v2]` attribute cannot be used on trait impl blocks
impl Trait for Enum {
    #[pin_v2] //~ ERROR `#[pin_v2]` attribute cannot be used on trait methods in impl blocks
    fn method() {}
}

#[pin_v2] //~ ERROR `#[pin_v2]` attribute cannot be used on extern crates
extern crate alloc;

#[pin_v2] //~ ERROR `#[pin_v2]` attribute cannot be used on use statements
use std::pin::Pin;

#[pin_v2] //~ ERROR `#[pin_v2]` attribute cannot be used on statics
static STATIC: () = ();

#[pin_v2] //~ ERROR `#[pin_v2]` attribute cannot be used on constants
const CONST: () = ();

#[pin_v2] //~ ERROR `#[pin_v2]` attribute cannot be used on functions
fn f<T, U>(#[pin_v2] param: Foo<T, U>)
//~^ ERROR `#[pin_v2]` attribute cannot be used on function params
//~| ERROR allow, cfg, cfg_attr, deny, expect, forbid, and warn are the only allowed built-in attributes in function parameters
where
    #[pin_v2]
    //~^ ERROR `#[pin_v2]` attribute cannot be used on where predicates
    T:,
{
    #[pin_v2] //~ ERROR `#[pin_v2]` attribute cannot be used on closures
    || ();
    #[pin_v2] //~ ERROR `#[pin_v2]` attribute cannot be used on expressions
    [(), (), ()];
    #[pin_v2] //~ ERROR `#[pin_v2]` attribute cannot be used on statements
    let _: Foo<(), ()> = Foo::StructVariant {
        #[pin_v2] //~ ERROR `#[pin_v2]` attribute cannot be used on struct fields
        field: (),
    };
    match param {
        #[pin_v2] //~ ERROR `#[pin_v2]` attribute cannot be used on match arms
        Foo::UnitVariant => {}
        Foo::TupleVariant(..) => {}
        Foo::StructVariant {
            #[pin_v2] //~ ERROR `#[pin_v2]` attribute cannot be used on pattern fields
            field,
        } => {}
    }
}

#[pin_v2] //~ ERROR `#[pin_v2]` attribute cannot be used on modules
mod m {}

#[pin_v2] //~ ERROR `#[pin_v2]` attribute cannot be used on foreign modules
extern "C" {
    #[pin_v2] //~ ERROR `#[pin_v2]` attribute cannot be used on foreign types
    type ForeignTy;

    #[pin_v2] //~ ERROR `#[pin_v2]` attribute cannot be used on foreign statics
    static EXTERN_STATIC: ();

    #[pin_v2] //~ ERROR `#[pin_v2]` attribute cannot be used on foreign functions
    fn extern_fn();
}

#[pin_v2] //~ ERROR `#[pin_v2]` attribute cannot be used on type alias
type Type = ();

#[pin_v2] //~ ERROR `#[pin_v2]` attribute cannot be used on macro defs
macro_rules! macro_def {
    () => {};
}

#[pin_v2] //~ ERROR `#[pin_v2]` attribute cannot be used on macro calls
macro_def!();

fn main() {}
