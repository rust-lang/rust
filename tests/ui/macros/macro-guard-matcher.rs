#![feature(macro_guard_matcher)]

fn main() {
    macro_rules! m {
        ($x:guard) => {};
    }

    // Accepts
    m!(if true);
    m!(if let Some(x) = Some(1));
    m!(if let Some(x) = Some(1) && x == 1);
    m!(if let Some(x) = Some(Some(1)) && let Some(1) = x);
    m!(if let Some(x) = Some(Some(1)) && let Some(y) = x && y == 1);

    // Rejects
    m!(let Some(x) = Some(1)); //~ERROR no rules expected keyword `let`
}
