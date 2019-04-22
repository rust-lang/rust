// compile-pass

#![allow(const_err)]

fn main() {
    let x: &'static _ = &|| { let z = 3; z };
}
