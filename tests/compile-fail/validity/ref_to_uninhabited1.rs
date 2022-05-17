#![feature(never_type)]
use std::mem::transmute;

fn main() { unsafe {
    let _x: &! = transmute(&42); //~ERROR encountered a reference pointing to uninhabited type !
} }
