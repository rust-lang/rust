
pure fn id<@T>(x: T) -> T { x }

fn unreachable() -> ! {
    fail "Internal error: entered unreachable code";
}

/* FIXME (issue #141):  See test/run-pass/constrained-type.rs.  Uncomment
 * the constraint once fixed. */
type rational = {num: int, den: int}; // : int::positive(*.den);

// : int::positive(*.den);
pure fn rational_leq(x: rational, y: rational) -> bool {
    // NB: Uses the fact that rationals have positive denominators WLOG:

    x.num * y.den <= y.num * x.den
}

pure fn orb(a: bool, b: bool) -> bool { a || b }

tag void {
    void(@void);
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
