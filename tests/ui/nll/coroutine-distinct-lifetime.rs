#![feature(coroutines)]

// Test for issue #47189. Here, both `s` and `t` are live for the
// coroutine's lifetime, but within the coroutine they have distinct
// lifetimes. We accept this code -- even though the borrow extends
// over a yield -- because the data that is borrowed (`*x`) is not
// stored on the stack.

//@ check-pass

fn foo(x: &mut u32) {
    #[coroutine]
    move || {
        let s = &mut *x;
        yield;
        *s += 1;

        let t = &mut *x;
        yield;
        *t += 1;
    };
}

fn main() {
    foo(&mut 0);
}
