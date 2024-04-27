#![warn(clippy::redundant_pattern_matching)]
#![allow(clippy::needless_if, clippy::no_effect, clippy::nonminimal_bool)]

macro_rules! condition {
    () => {
        true
    };
}

macro_rules! lettrue {
    (if) => {
        if let true = true {}
    };
    (while) => {
        while let true = true {}
    };
}

fn main() {
    let mut k = 5;

    if let true = k > 1 {}
    if let false = k > 5 {}
    if let (true) = k > 1 {}
    if let (true, true) = (k > 1, k > 2) {}
    while let true = k > 1 {
        k += 1;
    }
    while let true = condition!() {
        k += 1;
    }

    matches!(k > 5, true);
    matches!(k > 5, false);
    // Whole loop is from a macro expansion, don't lint:
    lettrue!(if);
    lettrue!(while);
}
