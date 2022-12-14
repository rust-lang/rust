// check-fail

#![feature(const_precise_live_drops)]

struct S;

impl Drop for S {
    fn drop(&mut self) {
        println!("Hello!");
    }
}

const fn foo() {
    let s = S; //~ destructor
}

fn main() {}
