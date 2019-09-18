#![feature(raw_ref_op)]

const A: () = { let mut x = 2; &raw mut x; };           //~ ERROR `&raw mut` is not allowed

static B: () = { let mut x = 2; &raw mut x; };          //~ ERROR `&raw mut` is not allowed

static mut C: () = { let mut x = 2; &raw mut x; };      //~ ERROR `&raw mut` is not allowed

const fn foo() {
    let mut x = 0;
    let y = &raw mut x;                                 //~ ERROR `&raw mut` is not allowed
}

fn main() {}
