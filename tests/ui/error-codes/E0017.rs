static X: i32 = 1;
const C: i32 = 2;
static mut M: i32 = 3;

const CR: &'static mut i32 = &mut C; //~ ERROR mutable references are not allowed
                                     //~| WARN taking a mutable
static STATIC_REF: &'static mut i32 = &mut X; //~ ERROR E0658
                                              //~| ERROR cannot borrow
                                              //~| ERROR mutable references are not allowed

static CONST_REF: &'static mut i32 = &mut C; //~ ERROR mutable references are not allowed
                                              //~| WARN taking a mutable
static STATIC_MUT_REF: &'static mut i32 = unsafe { &mut M }; //~ ERROR mutable references are not
fn main() {}
