// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;
use proc_macro::*;

#[proc_macro]
pub fn my_macro(input: TokenStream) -> TokenStream {
    input
}

#[proc_macro_attribute]
pub fn my_macro_attr(input: TokenStream, _: TokenStream) -> TokenStream {
    input
}

#[proc_macro_derive(MyTrait)]
pub fn my_macro_derive(input: TokenStream) -> TokenStream {
    input
}

fn check_bang1() {
    my_macro!(); //~ ERROR can't use a procedural macro from the same crate that defines it
}
fn check_bang2() {
    my_macro_attr!(); //~ ERROR cannot find macro `my_macro_attr!` in this scope
}
fn check_bang3() {
    MyTrait!(); //~ ERROR cannot find macro `MyTrait!` in this scope
}

#[my_macro] //~ ERROR attribute `my_macro` is currently unknown
fn check_attr1() {}
#[my_macro_attr] //~ ERROR can't use a procedural macro from the same crate that defines it
fn check_attr2() {}
#[MyTrait] //~ ERROR can't use a procedural macro from the same crate that defines it
           //~| ERROR `MyTrait` is a derive macro
fn check_attr3() {}

#[derive(my_macro)] //~ ERROR cannot find derive macro `my_macro` in this scope
struct CheckDerive1;
#[derive(my_macro_attr)] //~ ERROR can't use a procedural macro from the same crate that defines it
                         //~| ERROR macro `my_macro_attr` may not be used for derive attributes
struct CheckDerive2;
#[derive(MyTrait)] //~ ERROR can't use a procedural macro from the same crate that defines it
struct CheckDerive3;
