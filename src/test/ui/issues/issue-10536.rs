// We only want to assert that this doesn't ICE, we don't particularly care
// about whether it nor it fails to compile.

macro_rules! foo{
    () => {{
        macro_rules! bar{() => (())}
        1
    }}
}

pub fn main() {
    foo!();

    assert!({one! two()}); //~ ERROR expected open delimiter

    // regardless of whether nested macro_rules works, the following should at
    // least throw a conventional error.
    assert!({one! two}); //~ ERROR expected open delimiter
}
