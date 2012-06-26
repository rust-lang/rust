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
    fn times(f: fn(uint)) {
        let mut c = 0u;
        while c < self { f(c); c += 1u; }
    }
}

impl util<T> for ~[T] {
    fn length_() -> uint { vec::len(self) }
    fn iter_(f: fn(T)) { for self.each {|x| f(x); } }
    fn map_<U>(f: fn(T) -> U) -> ~[U] {
        let mut r = ~[];
        for self.each {|elt| r += ~[f(elt)]; }
        r
    }
}

fn main() {
    impl foo for int { fn plus() -> int { self + 10 } }
    assert 10.plus() == 20;
    assert 10u.plus() == 30;
    assert "hi".plus() == 200;

    assert (~[1]).length_().str() == "1";
    assert (~[3, 4]).map_({|a| a + 4})[0] == 7;
    assert (~[3, 4]).map_::<uint>({|a| a as uint + 4u})[0] == 7u;
    let mut x = 0u;
    10u.times({|_n| x += 2u;});
    assert x == 20u;
}
