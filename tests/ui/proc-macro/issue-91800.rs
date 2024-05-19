//@ aux-build: issue-91800-macro.rs

#[macro_use]
extern crate issue_91800_macro;

#[derive(MyTrait)]
//~^ ERROR macros that expand to items must be delimited with braces or followed by a semicolon
//~| ERROR proc-macro derive produced unparsable tokens
#[attribute_macro]
//~^ ERROR macros that expand to items must be delimited with braces or followed by a semicolon
struct MyStruct;

fn_macro! {}
//~^ ERROR macros that expand to items must be delimited with braces or followed by a semicolon

fn main() {}
