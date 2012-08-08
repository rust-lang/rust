trait double {
    fn double() -> uint;
}

impl uint: double {
    fn double() -> uint { self * 2u }
}

fn main() {
    let x = @3u;
    assert x.double() == 6u;
}
