#![feature(const_fn)]

struct S {
    state: u32,
}

impl S {
    const fn foo(&mut self, x: u32) {
        self.state = x;
    }
}

const FOO: S = {
    let mut s = S { state: 42 };
    s.foo(3);
    s
};
// The `impl` and `const` would error out if the following `type` was correct.
// See `const_let_assignment3.rs`.

type Array = [u32; {
    let mut x = 2;
    let y = &mut x;
    //~^ ERROR mutable references are not allowed in constants
    *y = 42;
    *y
}];

fn main() {
    assert_eq!(FOO.state, 3);
}
