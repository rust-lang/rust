// run-pass
// compile-flags: --test

#![feature(rustc_main)]
#![rustc_main(a::c)]

#![allow(dead_code)]

mod a {
    fn c() { panic!(); }
}
