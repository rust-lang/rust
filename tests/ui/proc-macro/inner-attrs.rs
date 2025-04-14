// gate-test-custom_inner_attributes
//@ compile-flags: -Z span-debug
//@ proc-macro: test-macros.rs
//@ edition:2018

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

#[print_target_and_args(mod_first)]
#[print_target_and_args(mod_second)]
mod inline_mod {
    #![print_target_and_args(mod_third)]
    #![print_target_and_args(mod_fourth)]
}

struct MyStruct {
    field: bool
}

#[derive(Print)]
struct MyDerivePrint {
    field: [u8; {
        match true {
            _ => {
                #![cfg_attr(not(FALSE), rustc_dummy(third))]
                true
            }
        };
        0
    }]
}

fn bar() {
    #[print_target_and_args(tuple_attrs)] (
        3, 4, {
            #![cfg_attr(not(FALSE), rustc_dummy(innermost))]
            5
        }
    );

    #[print_target_and_args(tuple_attrs)] (
        3, 4, {
            #![cfg_attr(not(FALSE), rustc_dummy(innermost))]
            5
        }
    );

    for _ in &[true] {
        #![print_attr]
        //~^ ERROR expected non-macro inner attribute, found attribute macro `print_attr`
    }

    let _ = {
        #![print_attr]
        //~^ ERROR expected non-macro inner attribute, found attribute macro `print_attr`
    };

    let _ = async {
        #![print_attr]
        //~^ ERROR expected non-macro inner attribute, found attribute macro `print_attr`
    };

    {
        #![print_attr]
        //~^ ERROR expected non-macro inner attribute, found attribute macro `print_attr`
    };
}


extern { //~ WARN extern declarations without an explicit ABI are deprecated
    fn weird_extern() {
        #![print_target_and_args_consume(tenth)]
    }
}

fn main() {}
