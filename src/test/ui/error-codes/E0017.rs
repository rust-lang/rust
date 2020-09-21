static X: i32 = 1;
const C: i32 = 2;
static mut M: i32 = 3;

const CR: &'static mut i32 = &mut C; //~ ERROR E0764
                                     //~| WARN taking a mutable
static STATIC_REF: &'static mut i32 = &mut X;
//~^ ERROR cannot borrow immutable static item `X` as mutable [E0596]
static CONST_REF: &'static mut i32 = &mut C; //~ ERROR E0764
                                              //~| WARN taking a mutable
static STATIC_MUT_REF: &'static mut i32 = unsafe { &mut M };
fn main() {}
