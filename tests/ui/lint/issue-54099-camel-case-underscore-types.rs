//@ check-pass

#![forbid(non_camel_case_types)]
#![allow(dead_code)]

// None of the following types should generate a warning
struct _X {}
struct __X {}
struct __ {}
struct X_ {}
struct X__ {}
struct X___ {}

fn main() { }
