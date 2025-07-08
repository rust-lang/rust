//@ aux-build: borrowck-error-in-macro.rs
//@ check-run-results
//FIXME: remove error-pattern (see #141896)

extern crate borrowck_error_in_macro as a;

a::ice! {}
//~^ ERROR cannot borrow value as mutable, as it is not declared as mutable
