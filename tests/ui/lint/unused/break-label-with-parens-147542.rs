//@ check-pass

// Regression test for #147542
// Ensures that we don't suggest removing parens in a break with label and loop
// when the parens are necessary for correct parsing.

#![warn(unused_parens)]
#![warn(break_with_label_and_loop)]

fn xyz() -> usize {
    'foo: {
        // parens bellow are necessary break of break with label and loop
        break 'foo ({
            println!("Hello!");
            123
        });
    }
}

fn main() {
    xyz();
}
