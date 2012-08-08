trait double {
    fn double() -> uint;
}

impl uint: double {
    fn double() -> uint { self * 2u }
}

enum foo = uint;

fn main() {
    let x = foo(3u);
    assert x.double() == 6u;
}
