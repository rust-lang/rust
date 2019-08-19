// run-pass
#![allow(unused_mut)]
#![allow(non_camel_case_types)]

enum int_wrapper<'a> {
    int_wrapper_ctor(&'a isize)
}

pub fn main() {
    let x = 3;
    let y = int_wrapper::int_wrapper_ctor(&x);
    let mut z : &isize;
    match y {
        int_wrapper::int_wrapper_ctor(zz) => { z = zz; }
    }
    println!("{}", *z);
}
