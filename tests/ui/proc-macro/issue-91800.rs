//@ proc-macro: issue-91800-macro.rs

#[macro_use]
extern crate issue_91800_macro;

#[derive(MyTrait)]
//~^ ERROR macros that expand to items must be delimited with braces or followed by a semicolon
//~| ERROR proc-macro derive produced unparsable tokens
//~| ERROR
#[attribute_macro]
//~^ ERROR macros that expand to items must be delimited with braces or followed by a semicolon
//~| ERROR
struct MyStruct;

fn_macro! {}
//~^ ERROR macros that expand to items must be delimited with braces or followed by a semicolon
//~| ERROR

fn main() {}
