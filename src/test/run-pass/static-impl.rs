use a::*;

trait plus {
    fn plus() -> int;
}

mod a {
    impl uint: plus { fn plus() -> int { self as int + 20 } }
}

mod b {
    impl ~str: plus { fn plus() -> int { 200 } }
}

trait uint_utils {
    fn str() -> ~str;
    fn multi(f: fn(uint));
}

impl uint: uint_utils {
    fn str() -> ~str { uint::str(self) }
    fn multi(f: fn(uint)) {
        let mut c = 0u;
        while c < self { f(c); c += 1u; }
    }
}

trait vec_utils<T> {
    fn length_() -> uint;
    fn iter_(f: fn(T));
    fn map_<U: copy>(f: fn(T) -> U) -> ~[U];
}

impl<T> ~[T]: vec_utils<T> {
    fn length_() -> uint { vec::len(self) }
    fn iter_(f: fn(T)) { for self.each |x| { f(x); } }
    fn map_<U: copy>(f: fn(T) -> U) -> ~[U] {
        let mut r = ~[];
        for self.each |elt| { r += ~[f(elt)]; }
        r
    }
}

fn main() {
    assert 10u.plus() == 30;
    assert (~"hi").plus() == 200;

    assert (~[1]).length_().str() == ~"1";
    assert (~[3, 4]).map_(|a| a + 4 )[0] == 7;
    assert (~[3, 4]).map_::<uint>(|a| a as uint + 4u )[0] == 7u;
    let mut x = 0u;
    10u.multi(|_n| x += 2u );
    assert x == 20u;
}
