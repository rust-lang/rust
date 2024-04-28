// Here we test all kinds of range patterns in terms of parsing / recovery.
// We want to ensure that:
// 1. Things parse as they should.
// 2. Or at least we have parser recovery if they don't.

#![deny(ellipsis_inclusive_range_patterns)]

fn main() {}

const X: u8 = 0;
const Y: u8 = 3;

fn exclusive_from_to() {
    if let 0..3 = 0 {} // OK.
    if let 0..Y = 0 {} // OK.
    if let X..3 = 0 {} // OK.
    if let X..Y = 0 {} // OK.
    if let true..Y = 0 {} //~ ERROR only `char` and numeric types
    if let X..true = 0 {} //~ ERROR only `char` and numeric types
    if let .0..Y = 0 {} //~ ERROR mismatched types
    //~^ ERROR float literals must have an integer part
    if let X.. .0 = 0 {} //~ ERROR mismatched types
    //~^ ERROR float literals must have an integer part
}

fn inclusive_from_to() {
    if let 0..=3 = 0 {} // OK.
    if let 0..=Y = 0 {} // OK.
    if let X..=3 = 0 {} // OK.
    if let X..=Y = 0 {} // OK.
    if let true..=Y = 0 {} //~ ERROR only `char` and numeric types
    if let X..=true = 0 {} //~ ERROR only `char` and numeric types
    if let .0..=Y = 0 {} //~ ERROR mismatched types
    //~^ ERROR float literals must have an integer part
    if let X..=.0 = 0 {} //~ ERROR mismatched types
    //~^ ERROR float literals must have an integer part
}

fn inclusive2_from_to() {
    if let 0...3 = 0 {}
    //~^ ERROR `...` range patterns are deprecated
    //~| WARN this is accepted in the current edition
    if let 0...Y = 0 {}
    //~^ ERROR `...` range patterns are deprecated
    //~| WARN this is accepted in the current edition
    if let X...3 = 0 {}
    //~^ ERROR `...` range patterns are deprecated
    //~| WARN this is accepted in the current edition
    if let X...Y = 0 {}
    //~^ ERROR `...` range patterns are deprecated
    //~| WARN this is accepted in the current edition
    if let true...Y = 0 {} //~ ERROR only `char` and numeric types
    //~^ ERROR `...` range patterns are deprecated
    //~| WARN this is accepted in the current edition
    if let X...true = 0 {} //~ ERROR only `char` and numeric types
    //~^ ERROR `...` range patterns are deprecated
    //~| WARN this is accepted in the current edition
    if let .0...Y = 0 {} //~ ERROR mismatched types
    //~^ ERROR float literals must have an integer part
    //~| WARN this is accepted in the current edition
    //~| ERROR `...` range patterns are deprecated
    if let X... .0 = 0 {} //~ ERROR mismatched types
    //~^ ERROR float literals must have an integer part
    //~| ERROR `...` range patterns are deprecated
    //~| WARN this is accepted in the current edition
}

fn exclusive_from() {
    if let 0.. = 0 {}
    if let X.. = 0 {}
    if let true.. = 0 {}
    //~^ ERROR only `char` and numeric types
    if let .0.. = 0 {}
    //~^ ERROR float literals must have an integer part
    //~| ERROR mismatched types
}

fn inclusive_from() {
    if let 0..= = 0 {} //~ ERROR inclusive range with no end
    if let X..= = 0 {} //~ ERROR inclusive range with no end
    if let true..= = 0 {} //~ ERROR inclusive range with no end
    //~| ERROR only `char` and numeric types
    if let .0..= = 0 {} //~ ERROR inclusive range with no end
    //~^ ERROR float literals must have an integer part
    //~| ERROR mismatched types
}

fn inclusive2_from() {
    if let 0... = 0 {} //~ ERROR inclusive range with no end
    if let X... = 0 {} //~ ERROR inclusive range with no end
    if let true... = 0 {} //~ ERROR inclusive range with no end
    //~| ERROR only `char` and numeric types
    if let .0... = 0 {} //~ ERROR inclusive range with no end
    //~^ ERROR float literals must have an integer part
    //~| ERROR mismatched types
}

fn exclusive_to() {
    if let ..0 = 0 {}
    if let ..Y = 0 {}
    if let ..true = 0 {}
    //~^ ERROR only `char` and numeric types
    if let .. .0 = 0 {}
    //~^ ERROR float literals must have an integer part
    //~| ERROR mismatched types
}

fn inclusive_to() {
    if let ..=3 = 0 {}
    if let ..=Y = 0 {}
    if let ..=true = 0 {}
    //~^ ERROR only `char` and numeric types
    if let ..=.0 = 0 {}
    //~^ ERROR float literals must have an integer part
    //~| ERROR mismatched types
}

fn inclusive2_to() {
    if let ...3 = 0 {}
    //~^ ERROR range-to patterns with `...` are not allowed
    if let ...Y = 0 {}
    //~^ ERROR range-to patterns with `...` are not allowed
    if let ...true = 0 {}
    //~^ ERROR range-to patterns with `...` are not allowed
    //~| ERROR only `char` and numeric types
    if let ....3 = 0 {}
    //~^ ERROR float literals must have an integer part
    //~| ERROR range-to patterns with `...` are not allowed
    //~| ERROR mismatched types
}

fn with_macro_expr_var() {
    macro_rules! mac2 {
        ($e1:expr, $e2:expr) => {
            let $e1..$e2;
            //~^ ERROR refutable pattern in local binding
            let $e1...$e2;
            //~^ ERROR `...` range patterns are deprecated
            //~| WARN this is accepted in the current edition
            //~| ERROR refutable pattern in local binding
            let $e1..=$e2;
            //~^ ERROR refutable pattern in local binding
        }
    }

    mac2!(0, 1);

    macro_rules! mac {
        ($e:expr) => {
            let ..$e;
            //~^ ERROR refutable pattern in local binding
            let ...$e;
            //~^ ERROR range-to patterns with `...` are not allowed
            //~| ERROR refutable pattern in local binding
            let ..=$e;
            //~^ ERROR refutable pattern in local binding
            let $e..;
            //~^ ERROR refutable pattern in local binding
            let $e...; //~ ERROR inclusive range with no end
            //~^ ERROR refutable pattern in local binding
            let $e..=; //~ ERROR inclusive range with no end
            //~^ ERROR refutable pattern in local binding
        }
    }

    mac!(0);
}
