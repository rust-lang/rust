#![feature(i128_type)]
#![allow(bad_style)]

extern crate compiler_builtins;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));
