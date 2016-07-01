#![feature(box_syntax)]

// FIXME: This span is wrong.
fn main() { //~ ERROR: tried to treat a memory pointer as a function pointer
    let x = box 42;
    unsafe {
        let f = std::mem::transmute::<Box<i32>, fn()>(x);
        f()
    }
}
