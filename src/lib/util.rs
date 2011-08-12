

fn id<T>(x: &T) -> T { ret x; }


/* FIXME (issue #141):  See test/run-pass/constrained-type.rs.  Uncomment
 * the constraint once fixed. */
type rational = {num: int, den: int};


// : int::positive(*.den);
fn rational_leq(x: &rational, y: &rational) -> bool {
    // NB: Uses the fact that rationals have positive denominators WLOG:

    ret x.num * y.den <= y.num * x.den;
}

fn orb(a: &bool, b: &bool) -> bool { ret a || b; }
// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
