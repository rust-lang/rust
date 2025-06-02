//@ aux-build: borrowck-error-in-macro.rs
//@ error-pattern: a call in this macro requires a mutable binding due to mutable borrow of `d`

extern crate borrowck_error_in_macro as a;

a::ice! {}
//~^ ERROR cannot borrow value as mutable, as it is not declared as mutable
