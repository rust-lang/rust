// Make sure that fn-to-block coercion isn't incorrectly lifted over
// other tycons.

fn main() {
    fn f(f: native fn(native fn(native fn()))) {
    }

    fn g(f: native fn(fn())) {
    }

    f(g);
    //~^ ERROR mismatched types: expected `extern fn(extern fn(extern fn()))`
}
