mod a {
    pub mod b {
        pub mod c {
            pub struct S;
            pub struct Z;
        }
    }
}

macro_rules! import {
    ($p: path) => (use $p {S, Z}); //~ERROR expected one of `::`, `;`, or `as`, found `{`
}

import! { a::b::c }

fn main() {}
