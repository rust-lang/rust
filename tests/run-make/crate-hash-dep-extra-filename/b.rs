extern crate a;

pub fn g() -> u32 {
    a::f()
}

pub fn make() -> a::S {
    a::S { field: 2 }
}
