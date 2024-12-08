//@ run-pass
// #42164

#![feature(decl_macro)]
#![allow(dead_code)]

pub macro m($inner_str:expr) {
    #[doc = $inner_str]
    struct S;
}

macro_rules! define_f {
    ($name:expr) => {
        #[export_name = $name]
        fn f() {}
    }
}

fn main() {
    define_f!(concat!("exported_", "f"));
    m!(stringify!(foo));
}
