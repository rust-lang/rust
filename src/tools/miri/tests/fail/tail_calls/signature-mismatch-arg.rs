#![feature(explicit_tail_calls)]
#![allow(incomplete_features)]

fn main() {
    // FIXME(explicit_tail_calls):
    //   the error should point to `become g(x)`,
    //   but tail calls mess up the backtrace it seems like...
    f(0);
    //~^ error: type i32 passing argument of type u32
}

fn f(x: u32) {
    let g = unsafe { std::mem::transmute::<fn(i32), fn(u32)>(g) };
    become g(x);
}

fn g(_: i32) {}
