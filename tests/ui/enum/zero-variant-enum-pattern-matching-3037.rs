// https://github.com/rust-lang/rust/issues/3037
//@ run-pass
#![allow(dead_code)]
#![allow(non_camel_case_types)]

enum what { }

fn what_to_string(x: what) -> String
{
    match x {
    }
}

pub fn main()
{
}
