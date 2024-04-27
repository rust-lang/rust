//@ check-pass

#![deny(non_camel_case_types)]

fn main() {}

trait foo_bar {
    #![allow(non_camel_case_types)]
}
