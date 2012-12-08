mod m {
    pub struct S {
        x: int,
        y: int
    }
}

fn main() {
    let x = m::S { x: 1, y: 2 };
    let m::S { x: a, y: b } = x;
}

