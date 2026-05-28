//@ proc-macro: three-equals.rs

extern crate three_equals;

use three_equals::three_equals;

fn main() {
    // This one is okay.
    three_equals!(===);

    // Need exactly three equals.
    three_equals!(==); //~ ERROR found 2 equal signs, need exactly 3

    // Need exactly three equals.
    three_equals!(=====); //~ ERROR expected EOF

    // Only equals accepted.
    three_equals!(abc); //~ ERROR expected `=`

    // Only equals accepted.
    three_equals!(!!); //~ ERROR expected `=`

    // Only three characters expected.
    three_equals!(===a); //~ ERROR expected EOF
}
