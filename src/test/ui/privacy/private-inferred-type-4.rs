#![feature(associated_consts)]
#![feature(decl_macro)]
#![allow(private_in_public)]

mod m {
    pub struct PubTupleStruct(u8);
    impl PubTupleStruct { fn method() {} }

    pub macro m() {
        PubTupleStruct;
        //~^ ERROR tuple struct `PubTupleStruct` is private
    }
}

fn main() {
    m::m!();
}
