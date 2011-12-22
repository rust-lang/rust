// error-pattern: mismatched types

// Make sure that fn-to-block coercion isn't incorrectly lifted over
// other tycons.

fn coerce(b: block()) -> fn() {
    fn lol(f: fn(block()) -> fn(), g: block()) -> fn() { ret f(g); }
    fn fn_id(f: fn()) -> fn() { ret f }
    ret lol(fn_id, b);
}


fn main() {
    let i = 8;
    let f = coerce(block () { log_full(core::error, i); });
    f(); }
