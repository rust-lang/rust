// Regression test for issue 90013.
//@ check-pass

fn main() {
    const { || {} };
}
