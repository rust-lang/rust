// Check that we expand multiple or-patterns from left to right.

// run-pass

#![feature(or_patterns)]

fn search(target: (bool, bool, bool)) -> u32 {
    let x = ((false, true), (false, true), (false, true));
    let mut guard_count = 0;
    match x {
        ((a, _) | (_, a), (b @ _, _) | (_, b @ _), (c @ false, _) | (_, c @ true))
            if {
                guard_count += 1;
                (a, b, c) == target
            } =>
        {
            guard_count
        }
        _ => unreachable!(),
    }
}

// Equivalent to the above code, but hopefully easier to understand.
fn search_old_style(target: (bool, bool, bool)) -> u32 {
    let x = ((false, true), (false, true), (false, true));
    let mut guard_count = 0;
    match x {
        ((a, _), (b @ _, _), (c @ false, _))
        | ((a, _), (b @ _, _), (_, c @ true))
        | ((a, _), (_, b @ _), (c @ false, _))
        | ((a, _), (_, b @ _), (_, c @ true))
        | ((_, a), (b @ _, _), (c @ false, _))
        | ((_, a), (b @ _, _), (_, c @ true))
        | ((_, a), (_, b @ _), (c @ false, _))
        | ((_, a), (_, b @ _), (_, c @ true))
            if {
                guard_count += 1;
                (a, b, c) == target
            } =>
        {
            guard_count
        }
        _ => unreachable!(),
    }
}

fn main() {
    assert_eq!(search((false, false, false)), 1);
    assert_eq!(search((false, false, true)), 2);
    assert_eq!(search((false, true, false)), 3);
    assert_eq!(search((false, true, true)), 4);
    assert_eq!(search((true, false, false)), 5);
    assert_eq!(search((true, false, true)), 6);
    assert_eq!(search((true, true, false)), 7);
    assert_eq!(search((true, true, true)), 8);

    assert_eq!(search_old_style((false, false, false)), 1);
    assert_eq!(search_old_style((false, false, true)), 2);
    assert_eq!(search_old_style((false, true, false)), 3);
    assert_eq!(search_old_style((false, true, true)), 4);
    assert_eq!(search_old_style((true, false, false)), 5);
    assert_eq!(search_old_style((true, false, true)), 6);
    assert_eq!(search_old_style((true, true, false)), 7);
    assert_eq!(search_old_style((true, true, true)), 8);
}
