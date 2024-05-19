//@ run-pass

fn main() {
    assert!(f("", 0));
    assert!(f("a", 1));
    assert!(f("b", 1));

    assert!(!f("", 1));
    assert!(!f("a", 0));
    assert!(!f("b", 0));

    assert!(!f("asdf", 32));

    ////

    assert!(!g(true, true, true));
    assert!(!g(false, true, true));
    assert!(!g(true, false, true));
    assert!(!g(false, false, true));
    assert!(!g(true, true, false));

    assert!(g(false, true, false));
    assert!(g(true, false, false));
    assert!(g(false, false, false));

    ////

    assert!(!h(true, true, true));
    assert!(!h(false, true, true));
    assert!(!h(true, false, true));
    assert!(!h(false, false, true));
    assert!(!h(true, true, false));

    assert!(h(false, true, false));
    assert!(h(true, false, false));
    assert!(h(false, false, false));
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
