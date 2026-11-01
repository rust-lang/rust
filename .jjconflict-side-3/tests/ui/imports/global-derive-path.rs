//@ edition: 2024
//@ check-pass
#![crate_type = "lib"]
#![feature(derive_macro_global_path)]

#[::core::derive(Clone)]
struct Y;

#[::std::derive(Clone)]
struct X;
