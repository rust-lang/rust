//@ run-pass
//! Test drop order for different ways of declaring pattern bindings involving or-patterns.
//! In particular, are ordered based on the order in which the first occurrence of each binding
//! appears (i.e. the "primary" bindings). Regression test for #142163.

use std::cell::RefCell;
use std::ops::Drop;

// For more informative failures, we collect drops in a `Vec` before checking their order.
struct DropOrder(RefCell<Vec<u32>>);
struct LogDrop<'o>(&'o DropOrder, u32);

impl<'o> Drop for LogDrop<'o> {
    fn drop(&mut self) {
        self.0.0.borrow_mut().push(self.1);
    }
}

#[track_caller]
fn assert_drop_order(expected_drops: impl IntoIterator<Item = u32>, f: impl Fn(&DropOrder)) {
    let order = DropOrder(RefCell::new(Vec::new()));
    f(&order);
    let order = order.0.into_inner();
    let correct_order: Vec<u32> = expected_drops.into_iter().collect();
    assert_eq!(order, correct_order);
}

#[expect(unused_variables, unused_assignments, irrefutable_let_patterns)]
fn main() {
    // When bindings are declared with `let pat;`, they're visited in left-to-right order, using the
    // order given by the first occurrence of each variable. They're later dropped in reverse.
    assert_drop_order(1..=3, |o| {
        // Drops are right-to-left: `z`, `y`, `x`.
        let (x, Ok(y) | Err(y), z);
        // Assignment order doesn't matter.
        z = LogDrop(o, 1);
        y = LogDrop(o, 2);
        x = LogDrop(o, 3);
    });
    assert_drop_order(1..=2, |o| {
        // The first or-pattern alternative determines the bindings' drop order: `y`, `x`.
        let ((true, x, y) | (false, y, x));
        x = LogDrop(o, 2);
        y = LogDrop(o, 1);
    });

    // `let pat = expr;` should have the same drop order.
    assert_drop_order(1..=3, |o| {
        // Drops are right-to-left: `z`, `y`, `x`.
        let (x, Ok(y) | Err(y), z) = (LogDrop(o, 3), Ok(LogDrop(o, 2)), LogDrop(o, 1));
    });
    assert_drop_order(1..=2, |o| {
        // The first or-pattern alternative determines the bindings' drop order: `y`, `x`.
        let ((true, x, y) | (false, y, x)) = (true, LogDrop(o, 2), LogDrop(o, 1));
    });
    assert_drop_order(1..=2, |o| {
        // That drop order is used regardless of which or-pattern alternative matches: `y`, `x`.
        let ((true, x, y) | (false, y, x)) = (false, LogDrop(o, 1), LogDrop(o, 2));
    });

    // `match` should have the same drop order.
    assert_drop_order(1..=3, |o| {
        // Drops are right-to-left: `z`, `y` `x`.
        match (LogDrop(o, 3), Ok(LogDrop(o, 2)), LogDrop(o, 1)) { (x, Ok(y) | Err(y), z) => {} }
    });
    assert_drop_order(1..=2, |o| {
        // The first or-pattern alternative determines the bindings' drop order: `y`, `x`.
        match (true, LogDrop(o, 2), LogDrop(o, 1)) { (true, x, y) | (false, y, x) => {} }
    });
    assert_drop_order(1..=2, |o| {
        // That drop order is used regardless of which or-pattern alternative matches: `y`, `x`.
        match (false, LogDrop(o, 1), LogDrop(o, 2)) { (true, x, y) | (false, y, x) => {} }
    });

    // Function params are visited one-by-one, and the order of bindings within a param's pattern is
    // the same as `let pat = expr;`
    assert_drop_order(1..=3, |o| {
        // Among separate params, the drop order is right-to-left: `z`, `y`, `x`.
        (|x, (Ok(y) | Err(y)), z| {})(LogDrop(o, 3), Ok(LogDrop(o, 2)), LogDrop(o, 1));
    });
    assert_drop_order(1..=3, |o| {
        // Within a param's pattern, likewise: `z`, `y`, `x`.
        (|(x, Ok(y) | Err(y), z)| {})((LogDrop(o, 3), Ok(LogDrop(o, 2)), LogDrop(o, 1)));
    });
    assert_drop_order(1..=2, |o| {
        // The first or-pattern alternative determines the bindings' drop order: `y`, `x`.
        (|((true, x, y) | (false, y, x))| {})((true, LogDrop(o, 2), LogDrop(o, 1)));
    });

    // `if let` and `let`-`else` see bindings in the same order as `let pat = expr;`.
    assert_drop_order(1..=3, |o| {
        if let (x, Ok(y) | Err(y), z) = (LogDrop(o, 3), Ok(LogDrop(o, 2)), LogDrop(o, 1)) {}
    });
    assert_drop_order(1..=3, |o| {
        let (x, Ok(y) | Err(y), z) = (LogDrop(o, 3), Ok(LogDrop(o, 2)), LogDrop(o, 1)) else {
            unreachable!();
        };
    });
    assert_drop_order(1..=2, |o| {
        if let (true, x, y) | (false, y, x) = (true, LogDrop(o, 2), LogDrop(o, 1)) {}
    });
    assert_drop_order(1..=2, |o| {
        let ((true, x, y) | (false, y, x)) = (true, LogDrop(o, 2), LogDrop(o, 1)) else {
            unreachable!();
        };
    });

    // Test nested and adjacent or-patterns, including or-patterns without bindings under a guard.
    assert_drop_order(1..=6, |o| {
        // The `LogDrop`s that aren't moved into bindings are dropped last.
        match [
            [LogDrop(o, 6), LogDrop(o, 4)],
            [LogDrop(o, 3), LogDrop(o, 2)],
            [LogDrop(o, 1), LogDrop(o, 5)],
        ] {
            [
                [_ | _, w | w] | [w | w, _ | _],
                [x | x, y | y] | [y | y, x | x],
                [z | z, _ | _] | [_ | _, z | z],
            ] if true => {}
            _ => unreachable!(),
        }
    });
}
