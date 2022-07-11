// Validation makes this fail in the wrong place
//@compile-flags: -Zmiri-disable-validation

#![feature(box_syntax)]

fn main() {
    let x = box 42;
    unsafe {
        let f = std::mem::transmute::<Box<i32>, fn()>(x);
        f() //~ ERROR: function pointer but it does not point to a function
    }
}
