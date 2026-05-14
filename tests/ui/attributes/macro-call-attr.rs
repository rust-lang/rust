//@ check-pass
#![feature(sanitize)]
#![feature(register_tool)]
#![feature(export_stable)]
#![feature(lang_items)]
#![feature(dropck_eyepatch)]
#![feature(diagnostic_on_const)]
#![feature(diagnostic_on_move)]
#![feature(diagnostic_on_unknown)]
#![feature(diagnostic_on_unmatch_args)]
#![warn(unused)]

macro_rules! test { () => {} }

#[doc = ""]
//~^ WARN unused doc comment
#[diagnostic::do_not_recommend]
//~^ WARN can only be placed on trait implementations
#[diagnostic::on_const]
//~^ WARN can only be applied to non-const trait implementations
#[diagnostic::on_move]
//~^ WARN can only be applied to enums, structs or unions
#[diagnostic::on_unimplemented]
//~^ WARN can only be applied to trait definitions
#[diagnostic::on_unknown]
//~^ WARN can only be applied to `use` statements
#[diagnostic::on_unmatch_args]
//~^ WARN can only be applied to macro definitions
#[sanitize()]
//~^ WARN attribute cannot be used on macro calls
//~| WARN previously accepted
#[register_tool(test)]
//~^ WARN attribute cannot be used on macro calls
//~| WARN previously accepted
#[link(name = "x")]
//~^ WARN attribute cannot be used on macro calls
//~| WARN previously accepted
#[export_stable]
//~^ WARN attribute cannot be used on macro calls
//~| WARN previously accepted
#[repr(align(64))]
//~^ WARN attribute cannot be used on macro calls
//~| WARN previously accepted
#[lang = "sized"]
//~^ WARN attribute cannot be used on macro calls
//~| WARN previously accepted
#[panic_handler]
//~^ WARN attribute cannot be used on macro calls
//~| WARN previously accepted
#[may_dangle]
//~^ WARN attribute cannot be used on macro calls
//~| WARN previously accepted
test!();

fn main() {}
