//@ run-pass

fn two_bindings(x: &((bool, bool), u8)) -> u8 {
    match x {
        &((true, y) | (y, true), z @ (0 | 4)) => (y as u8) + z,
        _ => 20,
    }
}

fn main() {
    assert_eq!(two_bindings(&((false, false), 0)), 20);
    assert_eq!(two_bindings(&((false, true), 0)), 0);
    assert_eq!(two_bindings(&((true, false), 0)), 0);
    assert_eq!(two_bindings(&((true, true), 0)), 1);
    assert_eq!(two_bindings(&((false, false), 4)), 20);
    assert_eq!(two_bindings(&((false, true), 4)), 4);
    assert_eq!(two_bindings(&((true, false), 4)), 4);
    assert_eq!(two_bindings(&((true, true), 4)), 5);
    assert_eq!(two_bindings(&((false, false), 3)), 20);
    assert_eq!(two_bindings(&((false, true), 3)), 20);
    assert_eq!(two_bindings(&((true, false), 3)), 20);
    assert_eq!(two_bindings(&((true, true), 3)), 20);
}
