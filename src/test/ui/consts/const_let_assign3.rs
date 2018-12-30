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

type Array = [u32; {
    let mut x = 2;
    let y = &mut x;
//~^ ERROR references in constants may only refer to immutable values
    *y = 42;
//~^ ERROR constant contains unimplemented expression type
    *y
}];

fn main() {
    assert_eq!(FOO.state, 3);
}
