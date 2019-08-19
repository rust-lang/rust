// build-pass (FIXME(62277): could be check-pass?)
#![allow(unused)]

macro_rules! column {
    ($i:ident) => {
        $i
    };
}

fn foo() -> ! {
    panic!();
}

fn main() {}
