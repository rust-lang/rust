static X: i32 = 1;
const C: i32 = 2;

const CR: &'static mut i32 = &mut C; //~ ERROR E0764
static STATIC_REF: &'static mut i32 = &mut X; //~ ERROR E0019
                                              //~| ERROR cannot borrow
                                              //~| ERROR E0764
static CONST_REF: &'static mut i32 = &mut C; //~ ERROR E0764

fn main() {}
