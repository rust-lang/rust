//@ check-pass

struct Foo;
impl Foo {
    fn and(self, _other: Foo) -> Foo { Foo }
}

fn example() {
    let mut interest = None;
    interest = match interest.take() { // expected: Option<?0>
        None => Some(Foo),
        Some(that) => Some(that.and(Foo)),
    }
}

fn main() {}
