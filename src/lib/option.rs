// lib/option::rs

tag t[T] {
    none;
    some(T);
}

type operator[T, U] = fn(&T) -> U;

fn get[T](&t[T] opt) -> T {
    alt (opt) {
        case (some[T](?x)) {
            ret x;
        }
        case (none[T]) {
            fail;
        }
    }
    fail;   // FIXME: remove me when exhaustiveness checking works
}

fn map[T, U](&operator[T, U] f, &t[T] opt) -> t[U] {
    alt (opt) {
        case (some[T](?x)) {
            ret some[U](f(x));
        }
        case (none[T]) {
            ret none[U];
        }
    }
    fail;   // FIXME: remove me when exhaustiveness checking works
}

fn is_none[T](&t[T] opt) -> bool {
    alt (opt) {
        case (none[T])      { ret true; }
        case (some[T](_))   { ret false; }
    }
}

fn from_maybe[T](&T def, &t[T] opt) -> T {
    auto f = bind util::id[T](_);
    ret maybe[T, T](def, f, opt);
}

fn maybe[T, U](&U def, fn(&T) -> U f, &t[T] opt) -> U {
    alt (opt) {
        case (none[T]) { ret def; }
        case (some[T](?t)) { ret f(t); }
    }
}

// Can be defined in terms of the above when/if we have const bind.
fn may[T](fn(&T) f, &t[T] opt) {
    alt (opt) {
        case (none[T]) { /* nothing */ }
        case (some[T](?t)) { f(t); }
    }
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C .. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:

