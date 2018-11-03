#![feature(box_syntax)]

fn main() {
    let x = box 42;
    unsafe {
        let _f = std::mem::transmute::<Box<i32>, fn()>(x); //~ ERROR encountered a pointer, but expected a function pointer
    }
}
