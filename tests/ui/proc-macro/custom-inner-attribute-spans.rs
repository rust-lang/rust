//@ build-pass
//@ proc-macro: custom-inner-attribute-spans.rs
//@ ignore-backends: gcc

#![feature(proc_macro_hygiene)]
#![feature(custom_inner_attributes)]

#[macro_use]
extern crate custom_inner_attribute_spans;

#[path = "auxiliary/custom-inner-attribute-spans-module.rs"]
mod tester;

fn main() {}
