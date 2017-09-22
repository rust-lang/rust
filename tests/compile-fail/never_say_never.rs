// This should fail even without validation
// compile-flags: -Zmir-emit-validate=0

#![feature(never_type)]
#![allow(unreachable_code)]

fn main() {
    let y = &5;
    let x: ! = unsafe {
        *(y as *const _ as *const !) //~ ERROR tried to access a dead local variable
    };
    f(x)
}

fn f(x: !) -> ! { x }
