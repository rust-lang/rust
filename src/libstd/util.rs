/*
Module: util
*/

/*
Function: id

The identity function
*/
pure fn id<T: copy>(x: T) -> T { x }

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

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
