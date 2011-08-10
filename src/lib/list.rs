import option::some;
import option::none;

tag list[T] { cons(T, @list[T]); nil; }

fn from_vec[@T](v: vec[T]) -> list[T] {
    let l = nil[T];
    // FIXME: This would be faster and more space efficient if it looped over
    // a reverse vector iterator. Unfortunately generic iterators seem not to
    // work yet.

    for item: T  in vec::reversed(v) { l = cons[T](item, @l); }
    ret l;
}

fn foldl[@T, @U](ls_: &list[T], u: &U, f: fn(&T, &U) -> U ) -> U {
    let accum: U = u;
    let ls = ls_;
    while true {
        alt ls {
          cons(hd, tl) { accum = f(hd, accum); ls = *tl; }
          nil. { break; }
        }
    }
    ret accum;
}

fn find[@T, @U](ls_: &list[T], f: fn(&T) -> option::t[U] ) -> option::t[U] {
    let ls = ls_;
    while true {
        alt ls {
          cons(hd, tl) {
            alt f(hd) { none. { ls = *tl; } some(rs) { ret some(rs); } }
          }
          nil. { break; }
        }
    }
    ret none;
}

fn has[@T](ls_: &list[T], elt: &T) -> bool {
    let ls = ls_;
    while true {
        alt ls {
          cons(hd, tl) { if elt == hd { ret true; } else { ls = *tl; } }
          nil. { break; }
        }
    }
    ret false;
}

fn length[@T](ls: &list[T]) -> uint {
    fn count[T](t: &T, u: &uint) -> uint { ret u + 1u; }
    ret foldl(ls, 0u, count);
}

fn cdr[@T](ls: &list[T]) -> list[T] {
    alt ls {
      cons(_, tl) { ret *tl; }
      nil. { fail "list empty" }
    }
}

fn car[@T](ls: &list[T]) -> T {
    alt ls {
      cons(hd, _) { ret hd; }
      nil. { fail "list empty" }
    }
}

fn append[@T](l: &list[T], m: &list[T]) -> list[T] {
    alt l {
      nil. { ret m; }
      cons(x, xs) {
        let rest = append(*xs, m);
        ret cons(x, @rest);
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
