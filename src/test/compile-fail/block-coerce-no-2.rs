// Make sure that fn-to-block coercion isn't incorrectly lifted over
// other tycons.

fn main() {
    fn f(f: extern fn(extern fn(extern fn()))) {
    }

    fn g(f: extern fn(fn())) {
    }

    f(g);
    //~^ ERROR mismatched types: expected `extern fn(extern fn(extern fn()))`
}
