#![feature(const_raw_ptr_deref)]

fn main() {}

// fine
const Z: i32 = unsafe { *(&1 as *const i32) };

// bad, will thus error in miri
const Z2: i32 = unsafe { *(42 as *const i32) }; //~ ERROR any use of this value will cause
//~| WARN this was previously accepted by the compiler but is being phased out
const Z3: i32 = unsafe { *(44 as *const i32) }; //~ ERROR any use of this value will cause
//~| WARN this was previously accepted by the compiler but is being phased out
