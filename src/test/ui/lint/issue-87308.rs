// Regression test for issue #87308.

// compile-flags: -Zunpretty=everybody_loops
// check-pass

macro_rules! foo {
    () => { break 'x; }
}

pub fn main() {
    'x: loop { foo!() }
}
