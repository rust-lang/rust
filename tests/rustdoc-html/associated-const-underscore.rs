#![feature(associated_const_underscore)]

pub struct Struct;

impl Struct {
    //@ has associated_const_underscore/struct.Struct.html '//*[@id="associatedconstant.NAMED"]' ''
    pub const NAMED: () = ();

    //@ !has associated_const_underscore/struct.Struct.html '//*[@id="associatedconstant._"]' ''
    pub const _: () = ();
}
