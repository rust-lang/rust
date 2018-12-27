#![feature(const_let)]
#![feature(const_fn)]

struct S {
    state: u32,
}

impl S {
    const fn foo(&mut self, x: u32) {
        self.state = x;
        //~^ contains unimplemented expression
    }
}

const FOO: S = {
    let mut s = S { state: 42 };
    s.foo(3); //~ ERROR references in constants may only refer to immutable values
    s
};

fn main() {
    assert_eq!(FOO.state, 3);
}
