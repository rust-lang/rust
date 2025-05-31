//@ normalize-stderr: "you are using [0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9]+)?( \([^)]*\))?" -> "you are using $$RUSTC_VERSION"

#![feature(doc_keyword)] //~ ERROR
#![feature(doc_primitive)] //~ ERROR
#![crate_type = "lib"]

pub fn foo() {}
