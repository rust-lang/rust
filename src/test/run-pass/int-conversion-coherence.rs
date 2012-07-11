// xfail-test
//
// Problem here is that transactions aren't implemented for integer literal
// inference.

trait plus {
    fn plus() -> int;
}

impl foo of plus for uint { fn plus() -> int { self as int + 20 } }
impl foo of plus for int { fn plus() -> int { self + 10 } }

fn main() {
    assert 10.plus() == 20;
}

