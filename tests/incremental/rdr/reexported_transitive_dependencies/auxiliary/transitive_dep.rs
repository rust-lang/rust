//@ compile-flags: -Z public-api-hash

#![crate_name = "transitive_dep"]
#![crate_type = "rlib"]

#[cfg(any(cpass1, cpass2))]
pub fn print() {
    println!("5");
}

#[cfg(any(cpass3))]
pub fn print() {
    println!("5");
}

#[cfg(any(cpass2))]
pub fn print2() {
    println!("5");
}
