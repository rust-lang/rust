// Suggest not mutably borrowing a mutable reference
#![crate_type = "rlib"]

pub fn f(b: &mut i32) {
    g(&mut b);
    //~^ ERROR cannot borrow
    //~| HELP try removing `&mut` here
    g(&mut &mut b);
    //~^ ERROR cannot borrow
    //~| HELP try removing `&mut` here
}

pub fn g(_: &mut i32) {}
