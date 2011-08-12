// lib/option::rs

tag t[@T] { none; some(T); }

fn get[@T](opt: &t[T]) -> T {
    alt opt {
      some(x) { x }
      none. { fail "option none" }
    }
}

fn map[@T, @U](f: &block(&T) -> U, opt: &t[T]) -> t[U] {
    alt opt { some(x) { some(f(x)) } none. { none } }
}

fn is_none[@T](opt: &t[T]) -> bool {
    alt opt { none. { true } some(_) { false } }
}

fn is_some[@T](opt: &t[T]) -> bool { !is_none(opt) }

fn from_maybe[@T](def: &T, opt: &t[T]) -> T {
    alt opt { some(x) { x } none. { def } }
}

fn maybe[@T, @U](def: &U, f: &block(&T) -> U, opt: &t[T]) -> U {
    alt opt { none. { def } some(t) { f(t) } }
}

// Can be defined in terms of the above when/if we have const bind.
fn may[@T](f: &block(&T), opt: &t[T]) {
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
