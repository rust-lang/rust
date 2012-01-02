/*
Module: util
*/

/*
Function: id

The identity function
*/
pure fn id<copy T>(x: T) -> T { x }

/*
Function: unreachable

A standard function to use to indicate unreachable code. Because the
function is guaranteed to fail typestate will correctly identify
any code paths following the appearance of this function as unreachable.
*/
fn unreachable() -> ! {
    fail "Internal error: entered unreachable code";
}

/* FIXME (issue #141):  See test/run-pass/constrained-type.rs.  Uncomment
 * the constraint once fixed. */
/*
Function: rational

A rational number
*/
type rational = {num: int, den: int}; // : int::positive(*.den);

/*
Function: rational_leq
*/
pure fn rational_leq(x: rational, y: rational) -> bool {
    // NB: Uses the fact that rationals have positive denominators WLOG:

    x.num * y.den <= y.num * x.den
}

/*
Function: orb
*/
pure fn orb(a: bool, b: bool) -> bool { a || b }

// FIXME: Document what this is for or delete it
tag void {
    void(@void);
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
