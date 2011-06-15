

fn id[T](&T x) -> T { ret x; }


/* FIXME (issue #141):  See test/run-pass/constrained-type.rs.  Uncomment
 * the constraint once fixed. */
type rational = rec(int num, int den);

 // : int::positive(*.den);
fn rational_leq(&rational x, &rational y) -> bool {
    // NB: Uses the fact that rationals have positive denominators WLOG:

    ret x.num * y.den <= y.num * x.den;
}

fn fst[T, U](&tup(T, U) x) -> T { ret x._0; }

fn snd[T, U](&tup(T, U) x) -> U { ret x._1; }

fn orb(&bool a, &bool b) -> bool { ret a || b; }
// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
