#![deny(unused_imports)]
#![allow(non_upper_case_globals)]

// The aim of this test is to ensure that deny/allow/warn directives
// are applied to individual "use" statements instead of silently
// ignored.

#[allow(dead_code)]
mod a { pub static x: isize = 3; pub static y: isize = 4; }

mod b {
    use a::x; //~ ERROR: unused import
    #[allow(unused_imports)]
    use a::y; // no error here
}

#[allow(unused_imports)]
mod c {
    use a::x;
    #[deny(unused_imports)]
    use a::y; //~ ERROR: unused import
}

fn main() {}
