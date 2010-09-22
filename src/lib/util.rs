tag option[T] {
    none;
    some(T);
}

type operator[T, U] = fn(&T) -> U;

fn option_map[T, U](&operator[T, U] f, &option[T] opt) -> option[U] {
    alt (opt) {
        case (some[T](?x)) {
            ret some[U](f(x));
        }
        case (none[T]) {
            ret none[U];
        }
    }
}

fn id[T](&T x) -> T {
    ret x;
}

/* FIXME (issue #141):  See test/run-pass/constrained-type.rs.  Uncomment
 * the constraint once fixed. */
type rational = rec(int num, int den); // : _int.positive(*.den);

fn rational_leq(&rational x, &rational y) -> bool {
    // NB: Uses the fact that rationals have positive denominators WLOG.
    ret x.num * y.den <= y.num * x.den;
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C .. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
