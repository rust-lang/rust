//@ run-pass

mod m {
    pub struct S {
        pub x: isize,
        pub y: isize
    }
}

pub fn main() {
    let x = m::S { x: 1, y: 2 };
    let m::S { x: _a, y: _b } = x;
}
