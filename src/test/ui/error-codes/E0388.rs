static X: i32 = 1;
const C: i32 = 2;

const CR: &'static mut i32 = &mut C; //~ ERROR E0658
static STATIC_REF: &'static mut i32 = &mut X; //~ ERROR E0658
                                              //~| ERROR cannot borrow
                                              //~| ERROR E0019
static CONST_REF: &'static mut i32 = &mut C; //~ ERROR E0658

fn main() {}
