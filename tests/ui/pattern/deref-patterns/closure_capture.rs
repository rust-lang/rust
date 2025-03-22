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

    let v = vec![1, 2, 3];
    let f = || {
        // this should count as a borrow of `v` as a whole
        let [.., x] = v else { unreachable!() };
        assert_eq!(x, 3);
    };
    assert_eq!(v, [1, 2, 3]);
    f();

    let mut b = Box::new("aaa".to_string());
    let mut f = || {
        let deref!(ref mut s) = b else { unreachable!() };
        s.push_str("aa");
    };
    f();
    assert_eq!(b.len(), 5);

    let mut v = vec![1, 2, 3];
    let mut f = || {
        // this should count as a mutable borrow of `v` as a whole
        let [.., ref mut x] = v else { unreachable!() };
        *x = 4;
    };
    f();
    assert_eq!(v, [1, 2, 4]);

    let mut v = vec![1, 2, 3];
    let mut f = || {
        // here, `[.., x]` is adjusted by both an overloaded deref and a builtin deref
        let [.., x] = &mut v else { unreachable!() };
        *x = 4;
    };
    f();
    assert_eq!(v, [1, 2, 4]);
}
