#![feature(raw_ref_op)]

const A: () = { let mut x = 2; &raw mut x; };           //~ mutable pointer

static B: () = { let mut x = 2; &raw mut x; };          //~ mutable pointer

const fn foo() {
    let mut x = 0;
    let y = &raw mut x;                                 //~ mutable pointer
}

fn main() {}
