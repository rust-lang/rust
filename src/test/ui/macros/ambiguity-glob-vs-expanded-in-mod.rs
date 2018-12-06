// compile-pass

#![feature(decl_macro)]

macro_rules! gen_mac { () => {
    pub macro mac() { () }
}}

mod m1 {
    pub macro mac() { 0 }
}

mod m2 {
    use m1::*;

    gen_mac!();
}

fn main() {
    m2::mac!() // OK
}
