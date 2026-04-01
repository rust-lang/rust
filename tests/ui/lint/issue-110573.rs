//@ check-pass

#![deny(warnings)]

pub struct Struct;

impl Struct {
    #[allow(non_upper_case_globals)]
    pub const Const: () = ();
}

fn main() {}
