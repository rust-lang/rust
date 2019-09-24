// run-pass
#![allow(dead_code)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
// ignore-pretty unreported

#![feature(box_syntax)]

pub trait bomb { fn boom(&self, _: Ident); }
pub struct S;
impl bomb for S { fn boom(&self, _: Ident) { } }

pub struct Ident { name: usize }

// macro_rules! int3 { () => ( unsafe { asm!( "int3" ); } ) }
macro_rules! int3 { () => ( { } ) }

fn Ident_new() -> Ident {
    int3!();
    Ident {name: 0x6789ABCD }
}

pub fn light_fuse(fld: Box<dyn bomb>) {
    int3!();
    let f = || {
        int3!();
        fld.boom(Ident_new()); // *** 1
    };
    f();
}

pub fn main() {
    let b = box S as Box<dyn bomb>;
    light_fuse(b);
}
