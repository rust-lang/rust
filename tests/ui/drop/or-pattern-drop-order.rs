//@ run-pass
//! Test drop order for different ways of declaring pattern bindings involving or-patterns.
//! Currently, it's inconsistent between language constructs (#142163).

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

    // When bindings are declared with `let pat = expr;`, bindings within or-patterns are seen last,
    // thus they're dropped first.
    assert_drop_order(1..=3, |o| {
        // Drops are right-to-left, treating `y` as rightmost: `y`, `z`, `x`.
        let (x, Ok(y) | Err(y), z) = (LogDrop(o, 3), Ok(LogDrop(o, 1)), LogDrop(o, 2));
    });
    assert_drop_order(1..=2, |o| {
        // The first or-pattern alternative determines the bindings' drop order: `y`, `x`.
        let ((true, x, y) | (false, y, x)) = (true, LogDrop(o, 2), LogDrop(o, 1));
    });
    assert_drop_order(1..=2, |o| {
        // That drop order is used regardless of which or-pattern alternative matches: `y`, `x`.
        let ((true, x, y) | (false, y, x)) = (false, LogDrop(o, 1), LogDrop(o, 2));
    });

    // `match` treats or-patterns as last like `let pat = expr;`, but also determines drop order
    // using the order of the bindings in the *last* or-pattern alternative.
    assert_drop_order(1..=3, |o| {
        // Drops are right-to-left, treating `y` as rightmost: `y`, `z`, `x`.
        match (LogDrop(o, 3), Ok(LogDrop(o, 1)), LogDrop(o, 2)) { (x, Ok(y) | Err(y), z) => {} }
    });
    assert_drop_order(1..=2, |o| {
        // The last or-pattern alternative determines the bindings' drop order: `x`, `y`.
        match (true, LogDrop(o, 1), LogDrop(o, 2)) { (true, x, y) | (false, y, x) => {} }
    });
    assert_drop_order(1..=2, |o| {
        // That drop order is used regardless of which or-pattern alternative matches: `x`, `y`.
        match (false, LogDrop(o, 2), LogDrop(o, 1)) { (true, x, y) | (false, y, x) => {} }
    });

    // Function params are visited one-by-one, and the order of bindings within a param's pattern is
    // the same as `let pat = expr`;
    assert_drop_order(1..=3, |o| {
        // Among separate params, the drop order is right-to-left: `z`, `y`, `x`.
        (|x, (Ok(y) | Err(y)), z| {})(LogDrop(o, 3), Ok(LogDrop(o, 2)), LogDrop(o, 1));
    });
    assert_drop_order(1..=3, |o| {
        // Within a param's pattern, or-patterns are treated as rightmost: `y`, `z`, `x`.
        (|(x, Ok(y) | Err(y), z)| {})((LogDrop(o, 3), Ok(LogDrop(o, 1)), LogDrop(o, 2)));
    });
    assert_drop_order(1..=2, |o| {
        // The first or-pattern alternative determines the bindings' drop order: `y`, `x`.
        (|((true, x, y) | (false, y, x))| {})((true, LogDrop(o, 2), LogDrop(o, 1)));
    });

    // `if let` and `let`-`else` see bindings in the same order as `let pat = expr;`.
    // Vars in or-patterns are seen last (dropped first), and the first alternative's order is used.
    assert_drop_order(1..=3, |o| {
        if let (x, Ok(y) | Err(y), z) = (LogDrop(o, 3), Ok(LogDrop(o, 1)), LogDrop(o, 2)) {}
    });
    assert_drop_order(1..=3, |o| {
        let (x, Ok(y) | Err(y), z) = (LogDrop(o, 3), Ok(LogDrop(o, 1)), LogDrop(o, 2)) else {
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
}
