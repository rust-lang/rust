// revisions: current next
//[next] compile-flags: -Ztrait-solver=next
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