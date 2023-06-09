// check-pass

#![allow(non_camel_case_types)]

struct union;

impl union {
    pub fn new() -> Self {
        union { }
    }
}

fn main() {
    let _u = union::new();
}
