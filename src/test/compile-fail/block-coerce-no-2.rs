// error-pattern: mismatched types

// Make sure that fn-to-block coercion isn't incorrectly lifted over
// other tycons.

fn main() {
    fn f(f: fn(fn(fn()))) {
    }

    fn g(f: fn(block())) {
    }

    f(g);
}
