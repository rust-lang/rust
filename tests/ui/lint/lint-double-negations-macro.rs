//@ check-pass
macro_rules! neg {
    ($e: expr) => {
        -$e
    };
}
macro_rules! bad_macro {
    ($e: expr) => {
        --$e //~ WARN use of a double negation
    };
}

fn main() {
    neg!(-1);
    bad_macro!(1);
}
