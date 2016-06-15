#![feature(box_syntax)]

fn main() {
    //FIXME: this span is wrong
    let x = box 42; //~ ERROR: tried to treat a memory pointer as a function pointer
    unsafe {
        let f = std::mem::transmute::<Box<i32>, fn()>(x);
        f()
    }
}
