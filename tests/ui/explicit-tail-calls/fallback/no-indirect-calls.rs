//@ compile-flags: -Zforce-tail-call-fallback
#![expect(incomplete_features)]
#![feature(explicit_tail_calls)]
#![crate_type = "lib"]

// We can't deduce the shim for a function pointer, and therefore cannot guarantee stack safety.

fn indirect_callee(x: i32) -> i32 {
    x
}

fn indirect_caller(x: i32) -> i32 {
    let f: fn(i32) -> i32 = indirect_callee;
    become f(x)
    //~^ ERROR `become` through a function pointer is not supported on this target
}
