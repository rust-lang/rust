//@ run-pass
//@ compile-flags: --test

#![feature(rustc_attrs)]

#![allow(dead_code)]

mod a {
    fn b() {
        (|| {
            #[rustc_main]
            fn c() { panic!(); }
        })();
    }
}
