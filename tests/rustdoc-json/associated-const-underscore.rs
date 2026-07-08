#![feature(associated_const_underscore)]

pub struct Struct;

impl Struct {
    //@ has "$..index[?(@.name=='NAMED')].inner.assoc_const"
    pub const NAMED: () = ();

    //@ !has "$..index[?(@.name=='_')]"
    pub const _: () = ();
}
