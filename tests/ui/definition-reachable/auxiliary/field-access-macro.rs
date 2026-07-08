#![feature(decl_macro)]

mod n {
    pub struct Struct(i32);
    pub fn get_struct() -> Struct { Struct(0) }

    pub macro allow_field_access($x:expr) {
        &mut $x.0
    }
}

pub use n::{allow_field_access, get_struct};

pub macro deny_field_access($x:expr) {
    &mut $x.0
}
