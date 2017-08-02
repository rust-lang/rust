// Validation makes this fail in the wrong place
// compile-flags: -Zmir-emit-validate=0

#![feature(box_syntax)]

fn main() {
    let x = box 42;
    unsafe {
        let f = std::mem::transmute::<Box<i32>, fn()>(x);
        f() //~ ERROR: tried to treat a memory pointer as a function pointer
    }
}
