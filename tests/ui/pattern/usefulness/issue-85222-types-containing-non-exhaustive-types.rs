struct A<T> {
    a: T,
}

struct B<T, U>(T, U);

fn main() {
    match 0 {
        //~^ ERROR non-exhaustive patterns: `usize::MAX..` not covered [E0004]
        0 => (),
        1..=usize::MAX => (),
    }

    match (0usize, 0usize) {
        //~^ ERROR non-exhaustive patterns: `(usize::MAX.., _)` not covered [E0004]
        (0, 0) => (),
        (1..=usize::MAX, 1..=usize::MAX) => (),
    }

    match (0isize, 0usize) {
        //~^ ERROR non-exhaustive patterns: `(..isize::MIN, _)` and `(isize::MAX.., _)` not covered [E0004]
        (isize::MIN..=isize::MAX, 0) => (),
        (isize::MIN..=isize::MAX, 1..=usize::MAX) => (),
    }

    // Should not report note about usize not having fixed max value
    match Some(1usize) {
        //~^ ERROR non-exhaustive patterns: `Some(_)` not covered
        None => {}
    }

    match Some(4) {
        //~^ ERROR non-exhaustive patterns: `Some(usize::MAX..)` not covered
        Some(0) => (),
        Some(1..=usize::MAX) => (),
        None => (),
    }

    match Some(Some(Some(0))) {
        //~^ ERROR non-exhaustive patterns: `Some(Some(Some(usize::MAX..)))` not covered
        Some(Some(Some(0))) => (),
        Some(Some(Some(1..=usize::MAX))) => (),
        Some(Some(None)) => (),
        Some(None) => (),
        None => (),
    }

    match (A { a: 0usize }) {
        //~^ ERROR non-exhaustive patterns: `A { a: usize::MAX.. }` not covered [E0004]
        A { a: 0 } => (),
        A { a: 1..=usize::MAX } => (),
    }

    match B(0isize, 0usize) {
        //~^ ERROR non-exhaustive patterns: `B(..isize::MIN, _)` and `B(isize::MAX.., _)` not covered [E0004]
        B(isize::MIN..=isize::MAX, 0) => (),
        B(isize::MIN..=isize::MAX, 1..=usize::MAX) => (),
    }

    // Should report only the note about usize not having fixed max value and not report
    // report the note about isize
    match B(0isize, 0usize) {
        //~^ ERROR non-exhaustive patterns: `B(_, usize::MAX..)` not covered [E0004]
        B(_, 0) => (),
        B(_, 1..=usize::MAX) => (),
    }
}
