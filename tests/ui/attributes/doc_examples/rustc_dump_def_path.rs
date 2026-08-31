//@ dont-require-annotations: ERROR
//@ compile-flags: --crate-type lib -Z ui-testing=no
//@ build-fail

#![feature(rustc_attrs)]

#[rustc_dump_def_path]
fn main() {
    || {
        unsafe extern "C" {
            #[rustc_dump_def_path]
            static Foo: u8;
        }
    };
}

mod a {
    mod b {
        mod c {
            #[rustc_dump_def_path]
            fn d() {}
        }
    }
}
