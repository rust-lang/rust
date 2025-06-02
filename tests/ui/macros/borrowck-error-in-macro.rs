//@ aux-build: borrowck-error-in-macro.rs

extern crate borrowck_error_in_macro as a;

a::ice! {}
