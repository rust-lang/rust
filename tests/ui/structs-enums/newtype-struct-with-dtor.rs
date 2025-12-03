//@ run-pass
#![allow(unconstructable_pub_struct)]

#![allow(unused_unsafe)]
#![allow(unused_variables)]

pub struct Fd(u32);

fn foo(a: u32) {}

impl Drop for Fd {
    fn drop(&mut self) {
        unsafe {
            let Fd(s) = *self;
            foo(s);
        }
    }
}

pub fn main() {
}
