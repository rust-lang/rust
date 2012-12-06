

// -*- rust -*-
extern mod std;

fn test1() {
    let mut s: ~str = ~"hello";
    s += ~"world";
    log(debug, copy s);
    assert (s[9] == 'd' as u8);
}

fn test2() {
    // This tests for issue #163

    let ff: ~str = ~"abc";
    let a: ~str = ff + ~"ABC" + ff;
    let b: ~str = ~"ABC" + ff + ~"ABC";
    log(debug, copy a);
    log(debug, copy b);
    assert (a == ~"abcABCabc");
    assert (b == ~"ABCabcABC");
}

fn main() { test1(); test2(); }
