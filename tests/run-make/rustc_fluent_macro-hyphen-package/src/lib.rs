#![feature(rustc_private)]

extern crate rustc_driver;
extern crate rustc_errors;
extern crate rustc_fluent_macro;

rustc_fluent_macro::fluent_messages!("../messages.ftl");
