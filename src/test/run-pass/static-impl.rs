import a::*;
import b::baz;

mod a {
    impl foo for uint { fn plus() -> int { self as int + 20 } }
}

mod b {
    impl baz for str { fn plus() -> int { 200 } }
}

fn main() {
    impl foo for int { fn plus() -> int { self + 10 } }
    assert 10.plus() == 20;
    assert 10u.plus() == 30;
    assert "hi".plus() == 200;
}

