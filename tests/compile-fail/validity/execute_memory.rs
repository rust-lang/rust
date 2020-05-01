#![feature(box_syntax)]

fn main() {
    let x = box 42;
    unsafe {
        let _f = std::mem::transmute::<Box<i32>, fn()>(x); //~ ERROR expected a function pointer
    }
}
