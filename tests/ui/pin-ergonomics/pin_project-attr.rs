#![feature(
    pin_ergonomics,
    where_clause_attrs,
    trait_alias,
    extern_types,
    associated_type_defaults,
    fn_delegation,
)]
#![allow(incomplete_features)]
#![pin_project] //~ ERROR `#[pin_project]` attribute cannot be used on crates

// allowed

#[pin_project]
struct Struct {}

#[pin_project]
enum Enum {}

#[pin_project]
union Union {
    field: (),
}

// disallowed

enum Foo<#[pin_project] T, #[pin_project] U = ()> {
    //~^ ERROR `#[pin_project]` attribute cannot be used on function params
    //~| ERROR `#[pin_project]` attribute cannot be used on function params
    #[pin_project] //~ ERROR `#[pin_project]` attribute cannot be used on enum variants
    UnitVariant,
    TupleVariant(#[pin_project] T), //~ ERROR `#[pin_project]` attribute cannot be used on struct fields
    StructVariant {
        #[pin_project] //~ ERROR `#[pin_project]` attribute cannot be used on struct fields
        field: U,
    },
}

#[pin_project] //~ ERROR `#[pin_project]` attribute cannot be used on traits
trait Trait {
    #[pin_project] //~ ERROR `#[pin_project]` attribute cannot be used on associated consts
    const ASSOC_CONST: () = ();
    #[pin_project] //~ ERROR `#[pin_project]` attribute cannot be used on associated types
    type AssocType = ();

    #[pin_project] //~ ERROR `#[pin_project]` attribute cannot be used on required trait methods
    fn method();
    #[pin_project] //~ ERROR `#[pin_project]` attribute cannot be used on provided trait methods
    fn method_with_body() {}
}

#[pin_project] //~ ERROR `#[pin_project]` attribute cannot be used on trait aliases
trait TraitAlias = Trait;

#[pin_project] //~ ERROR `#[pin_project]` attribute cannot be used on inherent impl blocks
impl Struct {
    // FIXME: delegation macros are not tested yet (how to?)
    #[pin_project] //~ ERROR `#[pin_project]` attribute cannot be used on delegations
    reuse <Struct as std::any::Any>::type_id;

    #[pin_project] //~ ERROR `#[pin_project]` attribute cannot be used on inherent methods
    fn method() {}
}

#[pin_project] //~ ERROR `#[pin_project]` attribute cannot be used on trait impl blocks
impl Trait for Enum {
    #[pin_project] //~ ERROR `#[pin_project]` attribute cannot be used on trait methods in impl blocks
    fn method() {}
}

#[pin_project] //~ ERROR `#[pin_project]` attribute cannot be used on extern crates
extern crate alloc;

#[pin_project] //~ ERROR `#[pin_project]` attribute cannot be used on use statements
use std::pin::Pin;

#[pin_project] //~ ERROR `#[pin_project]` attribute cannot be used on statics
static STATIC: () = ();

#[pin_project] //~ ERROR `#[pin_project]` attribute cannot be used on constants
const CONST: () = ();

#[pin_project] //~ ERROR `#[pin_project]` attribute cannot be used on functions
fn f<T, U>(#[pin_project] param: Foo<T, U>)
//~^ ERROR `#[pin_project]` attribute cannot be used on function params
//~| ERROR allow, cfg, cfg_attr, deny, expect, forbid, and warn are the only allowed built-in attributes in function parameters
where
    #[pin_project]
    //~^ ERROR `#[pin_project]` attribute cannot be used on where predicates
    T:,
{
    #[pin_project] //~ ERROR `#[pin_project]` attribute cannot be used on closures
    || ();
    #[pin_project] //~ ERROR `#[pin_project]` attribute cannot be used on expressions
    [(), (), ()];
    #[pin_project] //~ ERROR `#[pin_project]` attribute cannot be used on statements
    let _: Foo<(), ()> = Foo::StructVariant {
        #[pin_project] //~ ERROR `#[pin_project]` attribute cannot be used on struct fields
        field: (),
    };
    match param {
        #[pin_project] //~ ERROR `#[pin_project]` attribute cannot be used on match arms
        Foo::UnitVariant => {}
        Foo::TupleVariant(..) => {}
        Foo::StructVariant {
            #[pin_project] //~ ERROR `#[pin_project]` attribute cannot be used on pattern fields
            field,
        } => {}
    }
}

#[pin_project] //~ ERROR `#[pin_project]` attribute cannot be used on modules
mod m {}

#[pin_project] //~ ERROR `#[pin_project]` attribute cannot be used on foreign modules
extern "C" {
    #[pin_project] //~ ERROR `#[pin_project]` attribute cannot be used on foreign types
    type ForeignTy;

    #[pin_project] //~ ERROR `#[pin_project]` attribute cannot be used on foreign statics
    static EXTERN_STATIC: ();

    #[pin_project] //~ ERROR `#[pin_project]` attribute cannot be used on foreign functions
    fn extern_fn();
}

#[pin_project] //~ ERROR `#[pin_project]` attribute cannot be used on type alias
type Type = ();

#[pin_project] //~ ERROR `#[pin_project]` attribute cannot be used on macro defs
macro_rules! macro_def {
    () => {};
}

#[pin_project] //~ ERROR `#[pin_project]` attribute cannot be used on macro calls
macro_def!();

std::arch::global_asm! {
    "{}",
    #[pin_project] //~ ERROR this attribute is not supported on assembly
    const 0
}

fn main() {}
