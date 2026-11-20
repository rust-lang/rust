//@ check-pass

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

type Array = [u32; {
    let mut x = 2;
    let y = &mut x;
    *y = 42;
    *y
}];

fn main() {
    assert_eq!(FOO.state, 3);
}
