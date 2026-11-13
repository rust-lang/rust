//@ compile-flags: --emit metadata --crate-type lib
//@ edition: 2018

#![crate_name = "foo"]

pub mod bar {
    pub use bool;
    pub use char as my_char;
}
