static X: i32 = 1;
const C: i32 = 2;

const CR: &'static mut i32 = &mut C; //~ ERROR mutable references are not allowed

//~| WARN taking a mutable
static STATIC_REF: &'static mut i32 = &mut X; //~ ERROR E0658

static CONST_REF: &'static mut i32 = &mut C; //~ ERROR mutable references are not allowed

//~| WARN taking a mutable

fn main() {}
