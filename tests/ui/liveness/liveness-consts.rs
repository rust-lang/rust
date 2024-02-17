//@ check-pass
#![warn(unused)]
#![allow(unreachable_code)]

pub static A: i32 = {
    let mut i = 0;
    let mut a = 0; //~ WARN variable `a` is assigned to, but never used
    while i < 10 {
        i += 1;
        a += 1;
    }
    i
};

pub const B: u32 = {
    let mut b = 1;
    b += 1; //~ WARN value assigned to `b` is never read
    b = 42;
    b
};

pub enum E {
    V1 = {
        let e = 1; //~ WARN unused variable: `e`
        1
    },
    V2 = {
        let _f = 10;
        2
    }
}

pub fn f(x: [u8; { let s = 17; 100 }]) -> [u8;  { let z = 18; 100 }] {
    //~^ WARN unused variable: `s`
    //~| WARN unused variable: `z`
    x
}

pub trait T {
    const T: usize = {
        let mut t = 10;
        t = t + t; //~ WARN value assigned to `t` is never read
        20
    };
}

impl T for String {
    const T: usize = {
        let w = 10; //~ WARN unused variable: `w`
        loop {
            break;
            let _ = w;
        }
        44
    };
}

fn main() {
    let _ = [(); {
        let z = 42; //~ WARN unused variable: `z`
        35
    }];
}
