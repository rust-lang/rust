


// lib/option::rs
tag t[T] { none; some(T); }

type operator[T, U] = fn(&T) -> U ;

fn get[T](&t[T] opt) -> T {
    ret alt (opt) { case (some(?x)) { x } case (none) { fail } };
}

fn map[T, U](&operator[T, U] f, &t[T] opt) -> t[U] {
    ret alt (opt) {
            case (some(?x)) { some[U](f(x)) }
            case (none) { none[U] }
        };
}

fn is_none[T](&t[T] opt) -> bool {
    ret alt (opt) { case (none) { true } case (some(_)) { false } };
}

fn from_maybe[T](&T def, &t[T] opt) -> T {
    auto f = bind util::id[T](_);
    ret maybe[T, T](def, f, opt);
}

fn maybe[T, U](&U def, fn(&T) -> U  f, &t[T] opt) -> U {
    ret alt (opt) { case (none) { def } case (some(?t)) { f(t) } };
}


// Can be defined in terms of the above when/if we have const bind.
fn may[T](fn(&T)  f, &t[T] opt) {
    alt (opt) { case (none) {/* nothing */ } case (some(?t)) { f(t); } }
}
// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
