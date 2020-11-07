// run-pass

#![feature(or_patterns)]

fn main() {
    assert_eq!(f("", 0), true);
    assert_eq!(f("a", 1), true);
    assert_eq!(f("b", 1), true);

    assert_eq!(f("", 1), false);
    assert_eq!(f("a", 0), false);
    assert_eq!(f("b", 0), false);

    assert_eq!(f("asdf", 032), false);

    ////

    assert_eq!(g(true, true, true), false);
    assert_eq!(g(false, true, true), false);
    assert_eq!(g(true, false, true), false);
    assert_eq!(g(false, false, true), false);
    assert_eq!(g(true, true, false), false);

    assert_eq!(g(false, true, false), true);
    assert_eq!(g(true, false, false), true);
    assert_eq!(g(false, false, false), true);

    ////

    assert_eq!(h(true, true, true), false);
    assert_eq!(h(false, true, true), false);
    assert_eq!(h(true, false, true), false);
    assert_eq!(h(false, false, true), false);
    assert_eq!(h(true, true, false), false);

    assert_eq!(h(false, true, false), true);
    assert_eq!(h(true, false, false), true);
    assert_eq!(h(false, false, false), true);
}

fn f(s: &str, num: usize) -> bool {
    match (s, num) {
        ("", 0) | ("a" | "b", 1) => true,

        _ => false,
    }
}

fn g(x: bool, y: bool, z: bool) -> bool {
    match (x, y, x, z) {
        (true | false, false, true, false) => true,
        (false, true | false, true | false, false) => true,
        (true | false, true | false, true | false, true) => false,
        (true, true | false, true | false, false) => false,
    }
}

fn h(x: bool, y: bool, z: bool) -> bool {
    match (x, (y, (x, (z,)))) {
        (true | false, (false, (true, (false,)))) => true,
        (false, (true | false, (true | false, (false,)))) => true,
        (true | false, (true | false, (true | false, (true,)))) => false,
        (true, (true | false, (true | false, (false,)))) => false,
    }
}
