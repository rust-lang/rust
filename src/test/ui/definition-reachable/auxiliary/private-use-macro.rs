#![feature(decl_macro)]

mod n {
    pub static S: i32 = 57;
}

use n::S;

pub macro m() {
    S
}
