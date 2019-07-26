#![feature(exclusive_range_pattern)]

// run-pass

fn main() {
    let incl_range = |x, b| {
        match x {
            0..=5 if b => 0,
            5..=10 if b => 1,
            1..=4 if !b => 2,
            _ => 3,
        }
    };
    assert_eq!(incl_range(3, false), 2);
    assert_eq!(incl_range(3, true), 0);
    assert_eq!(incl_range(5, false), 3);
    assert_eq!(incl_range(5, true), 0);

    let excl_range = |x, b| {
        match x {
            0..5 if b => 0,
            5..10 if b => 1,
            1..4 if !b => 2,
            _ => 3,
        }
    };
    assert_eq!(excl_range(3, false), 2);
    assert_eq!(excl_range(3, true), 0);
    assert_eq!(excl_range(5, false), 3);
    assert_eq!(excl_range(5, true), 1);

    let incl_range_vs_const = |x, b| {
        match x {
            0..=5 if b => 0,
            7 => 1,
            3 => 2,
            _ => 3,
        }
    };
    assert_eq!(incl_range_vs_const(5, false), 3);
    assert_eq!(incl_range_vs_const(5, true), 0);
    assert_eq!(incl_range_vs_const(3, false), 2);
    assert_eq!(incl_range_vs_const(3, true), 0);
    assert_eq!(incl_range_vs_const(7, false), 1);
    assert_eq!(incl_range_vs_const(7, true), 1);

    let excl_range_vs_const = |x, b| {
        match x {
            0..5 if b => 0,
            7 => 1,
            3 => 2,
            _ => 3,
        }
    };
    assert_eq!(excl_range_vs_const(5, false), 3);
    assert_eq!(excl_range_vs_const(5, true), 3);
    assert_eq!(excl_range_vs_const(3, false), 2);
    assert_eq!(excl_range_vs_const(3, true), 0);
    assert_eq!(excl_range_vs_const(7, false), 1);
    assert_eq!(excl_range_vs_const(7, true), 1);

    let const_vs_incl_range = |x, b| {
        match x {
            3 if b => 0,
            5..=7 => 2,
            1..=4 => 1,
            _ => 3,
        }
    };
    assert_eq!(const_vs_incl_range(3, false), 1);
    assert_eq!(const_vs_incl_range(3, true), 0);

    let const_vs_excl_range = |x, b| {
        match x {
            3 if b => 0,
            5..7 => 2,
            1..4 => 1,
            _ => 3,
        }
    };
    assert_eq!(const_vs_excl_range(3, false), 1);
    assert_eq!(const_vs_excl_range(3, true), 0);
}
