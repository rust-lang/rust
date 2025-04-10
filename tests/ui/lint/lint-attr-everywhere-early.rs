// Tests that lint levels can be set for early lints.
#![allow(non_camel_case_types, unsafe_code, while_true, unused_parens)]

// The following is a check of the lints used here to verify they do not warn
// when allowed.
fn verify_no_warnings() {
    type non_camel_type = i32; // non_camel_case_types
    struct NON_CAMEL_IS_ALLOWED; // non_camel_case_types
    unsafe {} // unsafe_code
    enum Enum {
        VARIANT_CAMEL // non_camel_case_types
    }
    fn generics<foo>() {} // non_camel_case_types
    while true {} // while_true
    type T = (i32); // unused_parens
}


// ################## Types

#[deny(non_camel_case_types)]
type type_outer = i32; //~ ERROR type `type_outer` should have an upper camel case name

type BareFnPtr = fn(#[deny(unused_parens)](i32)); //~ ERROR unnecessary parentheses around type
// There aren't any early lints that currently apply to the variadic spot.
// type BareFnPtrVariadic = extern "C" fn(i32, #[deny()]...);

// ################## Items
#[deny(non_camel_case_types)]
struct ITEM_OUTER; //~ ERROR type `ITEM_OUTER` should have an upper camel case name

mod module_inner {
    #![deny(unsafe_code)]
    fn f() {
        unsafe {} //~ ERROR usage of an `unsafe` block
    }
}

struct Associated;
impl Associated {
    #![deny(unsafe_code)]

    fn inherent_denied_from_inner() { unsafe {} } //~ ERROR usage of an `unsafe` block

    #[deny(while_true)]
    fn inherent_fn() { while true {} } //~ ERROR denote infinite loops with

    #[deny(while_true)]
    const INHERENT_CONST: i32 = {while true {} 1}; //~ ERROR denote infinite loops with
}

trait trait_inner { //~ ERROR trait `trait_inner` should have an upper camel case name
    #![deny(non_camel_case_types)]
}

trait AssociatedTrait {
    #![deny(unsafe_code)]

    fn denied_from_inner() { unsafe {} } //~ ERROR usage of an `unsafe` block

    #[deny(while_true)]
    fn assoc_fn() { while true {} } //~ ERROR denote infinite loops with

    #[deny(while_true)]
    const ASSOC_CONST: i32 = {while true {} 1}; //~ ERROR denote infinite loops with

    #[deny(non_camel_case_types)]
    type assoc_type; //~ ERROR associated type `assoc_type` should have an upper camel case name
}

impl AssociatedTrait for Associated {
    #![deny(unsafe_code)]

    fn denied_from_inner() { unsafe {} } //~ ERROR usage of an `unsafe` block

    #[deny(while_true)]
    fn assoc_fn() { while true {} } //~ ERROR denote infinite loops with

    #[deny(while_true)]
    const ASSOC_CONST: i32 = {while true {} 1};  //~ ERROR denote infinite loops with

    #[deny(unused_parens)]
    type assoc_type = (i32); //~ ERROR unnecessary parentheses around type
}

struct StructFields {
    #[deny(unused_parens)]f1: (i32), //~ ERROR unnecessary parentheses around type
}
struct StructTuple(#[deny(unused_parens)](i32)); //~ ERROR unnecessary parentheses around type

enum Enum {
    #[deny(non_camel_case_types)]
    VARIANT_CAMEL, //~ ERROR variant `VARIANT_CAMEL` should have an upper camel case name
}

extern "C" {
    #![deny(unused_parens)]

    fn foreign_denied_from_inner(x: (i32)); //~ ERROR unnecessary parentheses around type
}

extern "C" {
    #[deny(unused_parens)]
    fn foreign_denied_from_outer(x: (i32)); //~ ERROR unnecessary parentheses around type
}

fn function(#[deny(unused_parens)] param: (i32)) {} //~ ERROR unnecessary parentheses around type

fn generics<#[deny(non_camel_case_types)]foo>() {} //~ ERROR type parameter `foo` should have an upper camel case name


// ################## Statements
fn statements() {
    #[deny(unused_parens)]
    let x = (1); //~ ERROR unnecessary parentheses around assigned value
}


// ################## Expressions
fn expressions() {
    let closure = |#[deny(unused_parens)] param: (i32)| {}; //~ ERROR unnecessary parentheses around type

    struct Match{f1: i32}
    // Strangely unused_parens doesn't fire with {f1: (123)}
    let f = Match{#[deny(unused_parens)]f1: {(123)}}; //~ ERROR unnecessary parentheses around block return value

    match f {
        #![deny(unsafe_code)]

        #[deny(while_true)]
        Match{f1} => {
            unsafe {} //~ ERROR usage of an `unsafe` block
            while true {} //~ ERROR denote infinite loops with
        }
    }

    match f {
        #[deny(ellipsis_inclusive_range_patterns)]
        Match{f1: 0...100} => {}
        //~^ ERROR range patterns are deprecated
        //~| WARNING this is accepted in the current edition
        _ => {}
    }

    // Statement Block
    {
        #![deny(unsafe_code)]
        unsafe {} //~ ERROR usage of an `unsafe` block
    }
    let block_tail: () = {
        #[deny(unsafe_code)]
        unsafe {} //~ ERROR usage of an `unsafe` block
    };

    // Before expression as a statement.
    #[deny(unsafe_code)]
    unsafe {}; //~ ERROR usage of an `unsafe` block

    [#[deny(unsafe_code)] unsafe {123}]; //~ ERROR usage of an `unsafe` block
    (#[deny(unsafe_code)] unsafe {123},); //~ ERROR usage of an `unsafe` block
    fn call(p: i32) {}
    call(#[deny(unsafe_code)] unsafe {123}); //~ ERROR usage of an `unsafe` block
    struct TupleStruct(i32);
    TupleStruct(#[deny(unsafe_code)] unsafe {123}); //~ ERROR usage of an `unsafe` block
}


// ################## Patterns
fn patterns() {
    struct PatField{f1: i32, f2: i32};
    let f = PatField{f1: 1, f2: 2};
    match f {
        PatField {
            #[deny(ellipsis_inclusive_range_patterns)]
            f1: 0...100,
            //~^ ERROR range patterns are deprecated
            //~| WARNING this is accepted in the current edition
            ..
        } => {}
        _ => {}
    }
}

fn main() {}
