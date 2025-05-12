//@ compile-flags: -Zdeduplicate-diagnostics=yes
#![deny(unknown_lints)]
//~^ NOTE defined here
#![allow(rustdoc::missing_doc_code_examples)]
//~^ ERROR unknown lint
//~| NOTE lint is unstable
//~| NOTE see issue
//~| NOTE: this compiler was built on YYYY-MM-DD; consider upgrading it if it is out of date
