//@ check-pass

const A: *const i32 = &raw const *&2;
static B: () = { &raw const *&2; };
static mut C: *const i32 = &raw const *&2;
const D: () = { let x = 2; &raw const x; };
static E: () = { let x = 2; &raw const x; };
static mut F: () = { let x = 2; &raw const x; };

const fn const_ptr() {
    let x = 0;
    let ptr = &raw const x;
    let r = &x;
    let ptr2 = &raw const *r;
}

fn main() {}
