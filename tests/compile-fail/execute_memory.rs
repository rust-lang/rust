#![feature(custom_attribute, box_syntax)]
#![allow(dead_code, unused_attributes)]

#[miri_run]
fn deref_fn_ptr() {
    //FIXME: this span is wrong
    let x = box 42; //~ ERROR: tried to treat a memory pointer as a function pointer
    unsafe {
        let f = std::mem::transmute::<Box<i32>, fn()>(x);
        f()
    }
}

fn main() {}
