#![expect(incomplete_features)]
#![feature(c_variadic, explicit_tail_calls)]
#![allow(unused)]

unsafe extern "C" fn foo(mut ap: ...) -> u32 {
    ap.arg::<u32>()
}

extern "C" fn bar() -> u32 {
    unsafe { become foo(1, 2, 3) }
    //~^ ERROR c-variadic functions can't be tail-called
}

fn main() {}
