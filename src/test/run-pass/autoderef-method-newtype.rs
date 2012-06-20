impl methods for uint {
    fn double() -> uint { self * 2u }
}

enum foo = uint;

fn main() {
    let x = foo(3u);
    assert x.double() == 6u;
}
