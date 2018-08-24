// compile-flags: --test

#![feature(main)]

#![allow(dead_code)]

mod a {
    fn b() {
        || {
            #[main]
            fn c() { panic!(); }
        };
    }
}
