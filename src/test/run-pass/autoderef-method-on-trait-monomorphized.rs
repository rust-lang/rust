trait double {
    fn double() -> uint;
}

impl uint: double {
    fn double() -> uint { self * 2u }
}

fn is_equal<D: double>(x: @D, exp: uint) {
    assert x.double() == exp;
}

fn main() {
    let x = @(3u as double);
    is_equal(x, 6);
}
