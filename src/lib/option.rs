


// lib/option::rs
tag t[@T] { none; some(T); }

type operator[@T, @U] = fn(&T) -> U ;

fn get[@T](opt: &t[T]) -> T { ret alt opt { some(x) { x } none. { fail } }; }

fn map[@T, @U](f: &operator[T, U], opt: &t[T]) -> t[U] {
    ret alt opt { some(x) { some[U](f(x)) } none. { none[U] } };
}

fn is_none[@T](opt: &t[T]) -> bool {
    ret alt opt { none. { true } some(_) { false } };
}

fn is_some[@T](opt: &t[T]) -> bool { ret !is_none(opt); }

fn from_maybe[@T](def: &T, opt: &t[T]) -> T {
    let f = bind util::id[T](_);
    ret maybe[T, T](def, f, opt);
}

fn maybe[@T, @U](def: &U, f: fn(&T) -> U , opt: &t[T]) -> U {
    ret alt opt { none. { def } some(t) { f(t) } };
}


// Can be defined in terms of the above when/if we have const bind.
fn may[@T](f: fn(&T) , opt: &t[T]) {
    alt opt { none. {/* nothing */ } some(t) { f(t); } }
}
// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
