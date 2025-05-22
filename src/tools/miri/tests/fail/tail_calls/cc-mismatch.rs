//@error-in-other-file: Undefined Behavior: calling a function with calling convention "C" using calling convention "Rust"
#![feature(explicit_tail_calls)]
#![allow(incomplete_features)]

fn main() {
    let f = unsafe { std::mem::transmute::<extern "C" fn(), fn()>(f) };
    become f();
}

extern "C" fn f() {}
