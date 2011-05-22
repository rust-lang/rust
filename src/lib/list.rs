import option::some;
import option::none;

// FIXME: It would probably be more appealing to define this as
// type list[T] = rec(T hd, option[@list[T]] tl), but at the moment
// our recursion rules do not permit that.

tag list[T] {
    cons(T, @list[T]);
    nil;
}

fn from_vec[T](vec[T] v) -> list[T] {
    auto l = nil[T];
    // FIXME: This would be faster and more space efficient if it looped over
    // a reverse vector iterator. Unfortunately generic iterators seem not to
    // work yet.
    for (T item in vec::reversed(v)) {
        l = cons[T](item, @l);
    }
    ret l;
}

fn foldl[T,U](&list[T] ls, &U u, fn(&T t, &U u) -> U f) -> U {
    alt(ls) {
        case (cons[T](?hd, ?tl)) {
            auto u_ = f(hd, u);
            be foldl[T,U](*tl, u_, f);
        }
        case (nil[T]) {
            ret u;
        }
    }
}

fn find[T,U](&list[T] ls,
             (fn(&T) -> option::t[U]) f) -> option::t[U] {
    alt(ls) {
        case (cons[T](?hd, ?tl)) {
            alt (f(hd)) {
                case (none[U]) {
                    be find[T,U](*tl, f);
                }
                case (some[U](?res)) {
                    ret some[U](res);
                }
            }
        }
        case (nil[T]) {
            ret none[U];
        }
    }
}

fn length[T](&list[T] ls) -> uint {
    fn count[T](&T t, &uint u) -> uint {
        ret u + 1u;
    }
    ret foldl[T,uint](ls, 0u, bind count[T](_, _));
}

fn cdr[T](&list[T] ls) -> list[T] {
    alt (ls) {
        case (cons[T](_, ?tl)) {ret *tl;}
    }
}
fn car[T](&list[T] ls) -> T {
    alt (ls) {
        case (cons[T](?hd, _)) {ret hd;}
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
