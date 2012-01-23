// Make sure that fn-to-block coercion isn't incorrectly lifted over
// other tycons.

fn coerce(b: fn()) -> native fn() {
    fn lol(f: native fn(fn()) -> native fn(),
           g: fn()) -> native fn() { ret f(g); }
    fn fn_id(f: native fn()) -> native fn() { ret f }
    ret lol(fn_id, b);
    //!^ ERROR mismatched types: expected `native fn(fn()) -> native fn()`
}

fn main() {
    let i = 8;
    let f = coerce({|| log(error, i); });
    f();
}
