#![feature(const_fn)]

struct S {
    state: u32,
}

impl S {
    const fn foo(&mut self, x: u32) {
        //~^ ERROR mutable reference
        self.state = x;
    }
}

const FOO: S = {
    let mut s = S { state: 42 };
    s.foo(3); //~ ERROR mutable reference
    s
};

fn main() {
    assert_eq!(FOO.state, 3);
}
