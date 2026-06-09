#![crate_name = "interesting"]
#![crate_type = "rlib"]

extern crate opaque;

#[no_mangle]
#[inline(never)]
pub fn function_called_twice(c: char) {
    if c == '2' {
        // This branch is taken twice
        opaque::f1();
    } else {
        // This branch is never taken
        opaque::f2();
    }
}

#[no_mangle]
#[inline(never)]
pub fn function_called_42_times(c: char) {
    if c == 'a' {
        // This branch is taken 12 times
        opaque::f1();
    } else {
        if c == 'b' {
            // This branch is taken 28 times
            opaque::f2();
        } else {
            // This branch is taken 2 times
            opaque::f3();
        }
    }
}

#[no_mangle]
#[inline(never)]
pub fn function_called_never(_: char) {
    opaque::f1();
}
