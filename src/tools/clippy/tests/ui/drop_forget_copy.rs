#![warn(clippy::drop_copy, clippy::forget_copy)]
#![allow(clippy::toplevel_ref_arg, clippy::drop_ref, clippy::forget_ref, unused_mut)]

use std::mem::{drop, forget};
use std::vec::Vec;

#[derive(Copy, Clone)]
struct SomeStruct {}

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

    drop(s1);
    drop(s2);
    drop(s3);
    drop(s4);
    drop(s5);

    forget(s1);
    forget(s2);
    forget(s3);
    forget(s4);
    forget(s5);

    let a1 = AnotherStruct {
        x: 255,
        y: 0,
        z: vec![1, 2, 3],
    };
    let a2 = &a1;
    let mut a3 = a1.clone();
    let ref a4 = a1;
    let a5 = a1.clone();

    drop(a2);
    drop(a3);
    drop(a4);
    drop(a5);

    forget(a2);
    let a3 = &a1;
    forget(a3);
    forget(a4);
    let a5 = a1.clone();
    forget(a5);
}
