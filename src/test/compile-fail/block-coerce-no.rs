// error-pattern: mismatched types
// xfail-stage0

// Make sure that fn-to-block coercion isn't incorrectly lifted over
// other tycons.

fn coerce(&block() b) -> fn() {
    fn lol(&fn(&block()) -> fn() f, &block() g) -> fn() {
        ret f(g);
    }
    fn fn_id (&fn() f) -> fn() { ret f }
    ret lol(fn_id, b);
}


fn main() {
    auto i = 8;
    auto f = coerce(block() { log_err i; } );
    f();
}
