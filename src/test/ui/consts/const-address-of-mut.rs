#![feature(raw_ref_op)]

const A: () = { let mut x = 2; &raw mut x; };           //~ mutable reference

static B: () = { let mut x = 2; &raw mut x; };          //~ mutable reference

static mut C: () = { let mut x = 2; &raw mut x; };      //~ mutable reference

const fn foo() {
    let mut x = 0;
    let y = &raw mut x;                                 //~ mutable reference
}

fn main() {}
