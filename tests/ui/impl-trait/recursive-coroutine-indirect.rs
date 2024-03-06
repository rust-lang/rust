//@ revisions: current next
//@[next] compile-flags: -Znext-solver

//@[next] build-fail
// Deeply normalizing writeback results of opaques makes this into a post-mono error :(

#![feature(coroutines)]
#![allow(unconditional_recursion)]
fn coroutine_hold() -> impl Sized {
    move || { //~ ERROR recursion in a coroutine requires boxing
        let x = coroutine_hold();
        yield;
        x;
    }
}

fn main() {}
