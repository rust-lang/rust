
import option;
import option::{some, none};

tag t<T, U> { left(T); right(U); }

fn either<T, U,
          V>(f_left: block(T) -> V, f_right: block(U) -> V, value: t<T, U>) ->
   V {
    alt value { left(l) { f_left(l) } right(r) { f_right(r) } }
}

fn lefts<T, U>(eithers: [t<T, U>]) -> [T] {
    let result: [T] = [];
    for elt: t<T, U> in eithers {
        alt elt { left(l) { result += [l] } _ {/* fallthrough */ } }
    }
    ret result;
}

fn rights<T, U>(eithers: [t<T, U>]) -> [U] {
    let result: [U] = [];
    for elt: t<T, U> in eithers {
        alt elt { right(r) { result += [r] } _ {/* fallthrough */ } }
    }
    ret result;
}

fn partition<T, U>(eithers: [t<T, U>]) -> {lefts: [T], rights: [U]} {
    let lefts: [T] = [];
    let rights: [U] = [];
    for elt: t<T, U> in eithers {
        alt elt { left(l) { lefts += [l] } right(r) { rights += [r] } }
    }
    ret {lefts: lefts, rights: rights};
}
//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
