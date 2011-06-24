
import option::some;
import option::none;


// FIXME: It would probably be more appealing to define this as
// type list[T] = rec(T hd, option[@list[T]] tl), but at the moment
// our recursion rules do not permit that.
tag list[T] { cons(T, @list[T]); nil; }

fn from_vec[T](vec[T] v) -> list[T] {
    auto l = nil[T];
    // FIXME: This would be faster and more space efficient if it looped over
    // a reverse vector iterator. Unfortunately generic iterators seem not to
    // work yet.

    for (T item in vec::reversed(v)) { l = cons[T](item, @l); }
    ret l;
}

fn foldl[T, U](&list[T] ls_, &U u, fn(&T, &U) -> U  f) -> U {
    let U accum = u;
    auto ls = ls_;
    while (true) {
        alt (ls) {
            case (cons(?hd, ?tl)) { accum = f(hd, accum); ls = *tl; }
            case (nil) { break; }
        }
    }
    ret accum;
}

fn find[T, U](&list[T] ls_, fn(&T) -> option::t[U]  f) -> option::t[U] {
    auto ls = ls_;
    while (true) {
        alt (ls) {
            case (cons(?hd, ?tl)) {
                alt (f(hd)) {
                    case (none) { ls = *tl; }
                    case (some(?rs)) { ret some(rs); }
                }
            }
            case (nil) { break; }
        }
    }
    ret none;
}

fn has[T](&list[T] ls_, &T elt) -> bool {
    auto ls = ls_;
    while (true) {
        alt (ls) {
            case (cons(?hd, ?tl)) {
                if (elt == hd) { ret true; } else { ls = *tl; }
            }
            case (nil) { ret false; }
        }
    }
    ret false; // Typestate checker doesn't understand infinite loops

}

fn length[T](&list[T] ls) -> uint {
    fn count[T](&T t, &uint u) -> uint { ret u + 1u; }
    ret foldl[T, uint](ls, 0u, bind count[T](_, _));
}

fn cdr[T](&list[T] ls) -> list[T] {
    alt (ls) { case (cons(_, ?tl)) { ret *tl; } }
}

fn car[T](&list[T] ls) -> T { alt (ls) { case (cons(?hd, _)) { ret hd; } } }

fn append[T](&list[T] l, &list[T] m) -> list[T] {
    alt (l) {
        case (nil) { ret m; }
        case (cons(?x, ?xs)) {
            let list[T] rest = append[T](*xs, m);
            ret cons[T](x, @rest);
        }
    }
}
// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
