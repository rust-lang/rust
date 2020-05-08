#![allow(incomplete_features)]
#![feature(unsized_locals, unsized_fn_params)]

use std::fmt;

fn gen_foo() -> Box<fmt::Display> {
    Box::new(Box::new("foo"))
}

fn foo(x: fmt::Display) {
    assert_eq!(x.to_string(), "foo");
}

fn foo_indirect(x: fmt::Display) {
    foo(x);
}

fn main() {
    foo(*gen_foo());
    foo_indirect(*gen_foo());

    {
        let x: fmt::Display = *gen_foo();
        foo(x);
    }

    {
        let x: fmt::Display = *gen_foo();
        let y: fmt::Display = *gen_foo();
        foo(x);
        foo(y);
    }

    {
        let mut cnt: usize = 3;
        let x = loop {
            let x: fmt::Display = *gen_foo();
            if cnt == 0 {
                break x;
            } else {
                cnt -= 1;
            }
        };
        foo(x);
    }

    {
        let x: fmt::Display = *gen_foo();
        let x = if true { x } else { *gen_foo() };
        foo(x);
    }
}
