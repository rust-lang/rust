//@ edition:2024
#![crate_type = "lib"]
#![feature(macro_metavar_expr)]
#![allow(incomplete_features)]
#![feature(macro_fragment_fields)]
#![feature(macro_fragments_more)]

macro_rules! bad1 {
    ($f:fn) => { ${f.unknown_field} };
    //~^ ERROR: expression of type `fn` has no field `unknown_field`
}

bad1! { fn f() {} }

macro_rules! bad2 {
    ($f:fn) => { ${f.name.unknown_field} };
    //~^ ERROR: expression of type `ident` has no field `unknown_field`
}

bad2! { fn f() {} }

macro_rules! bad3 {
    ($f:fn) => { ${f.name.unknown_field.unknown_field} };
    //~^ ERROR: expression of type `ident` has no field `unknown_field`
}

bad3! { fn f() {} }

macro_rules! bad4 {
    ($f:item) => { ${f.name} };
    //~^ ERROR: expression of type `item` has no field `name`
}

bad4! { fn f() {} }

macro_rules! bad5 {
    ($f:block) => { ${f.unknown_field} };
    //~^ ERROR: expression of type `block` has no field `unknown_field`
}

bad5! { { 42; } }

macro_rules! bad6 {
    ($a:adt) => { ${a.unknown_field} };
    //~^ ERROR: expression of type `adt` has no field `unknown_field`
}

bad6! { struct S; }

macro_rules! bad7 {
    ($a:adt) => { ${a.name.unknown_field} };
    //~^ ERROR: expression of type `ident` has no field `unknown_field`
}

bad7! { struct S; }
