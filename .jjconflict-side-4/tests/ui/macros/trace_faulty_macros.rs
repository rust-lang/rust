//@ compile-flags: -Z trace-macros

#![recursion_limit = "4"]

macro_rules! my_faulty_macro {
    () => {
        my_faulty_macro!(bcd); //~ ERROR no rules
    };
}

macro_rules! pat_macro {
    () => {
        pat_macro!(A{a:a, b:0, c:_, ..});
    };
    ($a:pat) => {
        $a //~ ERROR expected expression
    };
}

macro_rules! my_recursive_macro {
    () => {
        my_recursive_macro!(); //~ ERROR recursion limit
    };
}

macro_rules! my_macro {
    () => {};
}

fn main() {
    my_faulty_macro!();
    my_recursive_macro!();
    test!();
    non_exisiting!();
    derive!(Debug);
    let a = pat_macro!();
}

#[my_macro]
fn use_bang_macro_as_attr() {}

#[derive(Debug)] //~ ERROR `derive` may only be applied to `struct`s
fn use_derive_macro_as_attr() {}

macro_rules! test {
    (let $p:pat = $e:expr) => {test!(($p,$e))};
    // this should be expr
    //           vvv
    (($p:pat, $e:pat)) => {let $p = $e;}; //~ ERROR expected expression, found `pat` metavariable
}

fn foo() {
    test!(let x = 1+1);
}
