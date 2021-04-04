// compile-flags: -Z span-debug --error-format human
// aux-build:test-macros.rs

#![feature(custom_inner_attributes)]
#![feature(proc_macro_hygiene)]
#![feature(stmt_expr_attributes)]
#![feature(rustc_attrs)]

#![no_std] // Don't load unnecessary hygiene information from std
extern crate std;

#[macro_use]
extern crate test_macros;

#[print_target_and_args(first)]
#[print_target_and_args(second)]
fn foo() {
    #![print_target_and_args(third)]
    #![print_target_and_args(fourth)]
}

struct MyStruct {
    field: bool
}

fn bar() {
    (#![print_target_and_args(fifth)] 1, 2);
    //~^ ERROR expected non-macro inner attribute, found attribute macro

    #[print_target_and_args(tuple_attrs)] (
        #![cfg_attr(FALSE, rustc_dummy)]
        3, 4, {
            #![cfg_attr(not(FALSE), rustc_dummy(innermost))]
            5
        }
    );

    #[print_target_and_args(array_attrs)] [
        #![rustc_dummy(inner)]
        true; 0
    ];

    [#![print_target_and_args(sixth)] 1 , 2];
    //~^ ERROR expected non-macro inner attribute, found attribute macro
    [#![print_target_and_args(seventh)] true ; 5];
    //~^ ERROR expected non-macro inner attribute, found attribute macro

    match 0 {
        #![print_target_and_args(eighth)]
        //~^ ERROR expected non-macro inner attribute, found attribute macro
        _ => {}
    }

    MyStruct { #![print_target_and_args(ninth)] field: true };
    //~^ ERROR expected non-macro inner attribute, found attribute macro
}

extern {
    fn weird_extern() {
        #![print_target_and_args_consume(tenth)]
    }
}

fn main() {}
