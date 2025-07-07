// We should not emit unused braces warning for `Mutex`
// See issue #143562
//@ build-pass
#![warn(unused)]
#![deny(warnings)]

use std::sync::Mutex;

struct Test {
    a: i32,
    b: i32,
}

struct VecVec {
    v: Vec<i32>,
}

impl VecVec {
    pub fn push(mut self, value: i32) -> Self {
        self.v.push(value);
        self
    }
}

pub fn main() {
    let test = Mutex::new(Test {
        a: 12,
        b: 34,
    });
    // the program will hang if this pair of braces is removed
    let vec_vec = VecVec { v: Vec::new() }
        .push({ test.lock().unwrap().a })
        .push({ test.lock().unwrap().b });
    println!("len: {}", vec_vec.v.len());
}
