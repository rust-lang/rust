static X: i32 = 1;
const C: i32 = 2;

const CR: &'static mut i32 = &mut C; //~ ERROR E0017
static STATIC_REF: &'static mut i32 = &mut X; //~ ERROR E0017
                                              //~| ERROR cannot borrow
                                              //~| ERROR cannot mutate statics
static CONST_REF: &'static mut i32 = &mut C; //~ ERROR E0017
fn main() {}
