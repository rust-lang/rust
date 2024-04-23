//@ run-pass
#![feature(deref_patterns)]
#![allow(incomplete_features)]

fn main() {
    let b = Box::new("aaa".to_string());
    let f = || {
        let deref!(ref s) = b else { unreachable!() };
        assert_eq!(s.len(), 3);
    };
    assert_eq!(b.len(), 3);
    f();

    let mut b = Box::new("aaa".to_string());
    let mut f = || {
        let deref!(ref mut s) = b else { unreachable!() };
        s.push_str("aa");
    };
    f();
    assert_eq!(b.len(), 5);
}
