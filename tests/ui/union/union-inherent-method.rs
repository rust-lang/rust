//@ run-pass

union U {
    a: u8,
}

impl U {
    fn method(&self) -> u8 { unsafe { self.a } }
}

fn main() {
    let u = U { a: 10 };
    assert_eq!(u.method(), 10);
}
