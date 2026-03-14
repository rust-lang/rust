// edition:2024
// compile-fail
// Regression test for issue #138710
// ICE: rustc panicked at compiler\rustc_middle\src\mir\interpret\queries.rs:104:13

#![feature(min_generic_const_args)]

trait B {
    type N: A;
}

trait A {
    const N: usize;
}

async fn fun() -> Box<dyn A> {
    *(&mut [0; <<Vec<u32> as B>::N as A>::N])
}

fn main() {}
