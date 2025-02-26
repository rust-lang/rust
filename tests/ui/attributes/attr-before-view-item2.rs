//@ build-pass (FIXME(62277): could be check-pass?)

#![feature(rustc_attrs)]
#![feature(test)]

mod m {
    #[rustc_dummy = "bar"]
    extern crate test;
}

fn main() {}
