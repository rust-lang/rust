//@ check-pass

#![warn(forgetting_copy_types)]

use std::mem::forget;
use std::vec::Vec;

#[derive(Copy, Clone)]
struct SomeStruct;

struct AnotherStruct {
    x: u8,
    y: u8,
    z: Vec<u8>,
}

impl Clone for AnotherStruct {
    fn clone(&self) -> AnotherStruct {
        AnotherStruct {
            x: self.x,
            y: self.y,
            z: self.z.clone(),
        }
    }
}

fn main() {
    let s1 = SomeStruct {};
    let s2 = s1;
    let s3 = &s1;
    let mut s4 = s1;
    let ref s5 = s1;

    forget(s1); //~ WARN calls to `std::mem::forget`
    forget(s2); //~ WARN calls to `std::mem::forget`
    forget(s3); //~ WARN calls to `std::mem::forget`
    forget(s4); //~ WARN calls to `std::mem::forget`
    forget(s5); //~ WARN calls to `std::mem::forget`

    let a1 = AnotherStruct {
        x: 255,
        y: 0,
        z: vec![1, 2, 3],
    };
    let a2 = &a1;
    let mut a3 = a1.clone();
    let ref a4 = a1;
    let a5 = a1.clone();

    forget(a2); //~ WARN calls to `std::mem::forget`
    let a3 = &a1;
    forget(a3); //~ WARN calls to `std::mem::forget`
    forget(a4); //~ WARN calls to `std::mem::forget`
    let a5 = a1.clone();
    forget(a5);
}
