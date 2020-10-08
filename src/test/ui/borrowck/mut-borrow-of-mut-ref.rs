// Suggest not mutably borrowing a mutable reference
#![crate_type = "rlib"]

pub fn f(b: &mut i32) {
    g(&mut b);
    //~^ ERROR cannot borrow
    g(&mut &mut b);
    //~^ ERROR cannot borrow
}

pub fn g(_: &mut i32) {}
