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
    //~^ redundant_pattern_matching
    if let false = k > 5 {}
    //~^ redundant_pattern_matching
    if let (true) = k > 1 {}
    //~^ redundant_pattern_matching
    if let (true, true) = (k > 1, k > 2) {}
    while let true = k > 1 {
        //~^ redundant_pattern_matching
        k += 1;
    }
    while let true = condition!() {
        //~^ redundant_pattern_matching
        k += 1;
    }

    matches!(k > 5, true);
    //~^ redundant_pattern_matching
    matches!(k > 5, false);
    //~^ redundant_pattern_matching
    // Whole loop is from a macro expansion, don't lint:
    lettrue!(if);
    lettrue!(while);
}
