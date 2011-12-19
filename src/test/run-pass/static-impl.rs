import a::*;
import b::baz;

mod a {
    impl foo for uint { fn plus() -> int { self as int + 20 } }
}

mod b {
    impl baz for str { fn plus() -> int { 200 } }
}

impl util for uint {
    fn str() -> str { uint::str(self) }
    fn times(f: block(uint)) {
        let c = 0u;
        while c < self { f(c); c += 1u; }
    }
}

impl util<T> for [T] {
    fn len() -> uint { vec::len(self) }
    fn iter(f: block(T)) { for x in self { f(x); } }
    fn map<U>(f: block(T) -> U) -> [U] {
        let r = [];
        for elt in self { r += [f(elt)]; }
        r
    }
}

fn main() {
    impl foo for int { fn plus() -> int { self + 10 } }
    assert 10.plus() == 20;
    assert 10u.plus() == 30;
    assert "hi".plus() == 200;

    assert [1].len().str() == "1";
    assert [3, 4].map({|a| a + 4})[0] == 7;
    assert [3, 4].map::<uint>({|a| a as uint + 4u})[0] == 7u;
    let x = 0u;
    10u.times {|_n| x += 2u;}
    assert x == 20u;
}
