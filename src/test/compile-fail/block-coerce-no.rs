// Make sure that fn-to-block coercion isn't incorrectly lifted over
// other tycons.

fn coerce(b: fn()) -> extern fn() {
    fn lol(f: extern fn(fn()) -> extern fn(),
           g: fn()) -> extern fn() { return f(g); }
    fn fn_id(f: extern fn()) -> extern fn() { return f }
    return lol(fn_id, b);
    //~^ ERROR mismatched types: expected `extern fn(fn()) -> extern fn()`
}

fn main() {
    let i = 8;
    let f = coerce(|| log(error, i) );
    f();
}
