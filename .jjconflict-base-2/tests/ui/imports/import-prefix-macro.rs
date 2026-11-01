//@ run-pass
#![allow(unused_variables)]
mod a {
    pub mod b {
        pub mod c {
            pub struct S;
            pub struct Z;
        }
        pub struct W;
    }
}

macro_rules! import {
    (1 $p: path) => (use $p;);
    (2 $p: path) => (use $p::{Z};);
    (3 $p: path) => (use $p::*;);
}

import! { 1 a::b::c::S }
import! { 2 a::b::c }
import! { 3 a::b }

fn main() {
    let s = S;
    let z = Z;
    let w = W;
}
