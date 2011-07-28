
import option::none;
import option::some;
import util::orb;

type vbuf = rustrt::vbuf;

type operator2[T, U, V] = fn(&T, &U) -> V ;

type array[T] = vec[mutable? T];

native "rust" mod rustrt {
    type vbuf;
    fn vec_buf[T](v: vec[T], offset: uint) -> vbuf;
    fn vec_len[T](v: vec[T]) -> uint;

    /**
     * Sometimes we modify the vec internal data via vec_buf and need to
     * update the vec's fill length accordingly.
     */
    fn vec_len_set[T](v: vec[T], n: uint);

    /**
     * The T in vec_alloc[T, U] is the type of the vec to allocate.  The
     * U is the type of an element in the vec.  So to allocate a vec[U] we
     * want to invoke this as vec_alloc[vec[U], U].
     */
    fn vec_alloc[T, U](n_elts: uint) -> vec[U];
    fn vec_alloc_mut[T, U](n_elts: uint) -> vec[mutable U];
    fn refcount[T](v: vec[T]) -> uint;
    fn vec_print_debug_info[T](v: vec[T]);
    fn vec_from_vbuf[T](v: vbuf, n_elts: uint) -> vec[T];
    fn unsafe_vec_to_mut[T](v: vec[T]) -> vec[mutable T];
}

fn alloc[T](n_elts: uint) -> vec[T] {
    ret rustrt::vec_alloc[vec[T], T](n_elts);
}

fn alloc_mut[T](n_elts: uint) -> vec[mutable T] {
    ret rustrt::vec_alloc_mut[vec[mutable T], T](n_elts);
}

fn refcount[T](v: array[T]) -> uint {
    let r = rustrt::refcount[T](v);
    if r == dbg::const_refcount { ret r; } else { ret r - 1u; }
}

fn vec_from_vbuf[T](v: vbuf, n_elts: uint) -> vec[T] {
    ret rustrt::vec_from_vbuf[T](v, n_elts);
}


// FIXME: Remove me; this is a botch to get around rustboot's bad typechecker.
fn empty[T]() -> vec[T] { ret alloc[T](0u); }


// FIXME: Remove me; this is a botch to get around rustboot's bad typechecker.
fn empty_mut[T]() -> vec[mutable T] { ret alloc_mut[T](0u); }

type init_op[T] = fn(uint) -> T ;

fn init_fn[@T](op: &init_op[T], n_elts: uint) -> vec[T] {
    let v: vec[T] = alloc[T](n_elts);
    let i: uint = 0u;
    while i < n_elts { v += [op(i)]; i += 1u; }
    ret v;
}

fn init_fn_mut[@T](op: &init_op[T], n_elts: uint) -> vec[mutable T] {
    let v: vec[mutable T] = alloc_mut[T](n_elts);
    let i: uint = 0u;
    while i < n_elts { v += [mutable op(i)]; i += 1u; }
    ret v;
}

// init_elt: creates and returns a vector of length n_elts, filled with
// that many copies of element t.
fn init_elt[@T](t: &T, n_elts: uint) -> vec[T] {
    /**
     * FIXME (issue #81): should be:
     *
     * fn elt_op[T](&T x, uint i) -> T { ret x; }
     * let init_op[T] inner = bind elt_op[T](t, _);
     * ret init_fn[T](inner, n_elts);
     */

    let v: vec[T] = alloc[T](n_elts);
    let i: uint = n_elts;
    while i > 0u { i -= 1u; v += [t]; }
    ret v;
}

fn init_elt_mut[@T](t: &T, n_elts: uint) -> vec[mutable T] {
    let v: vec[mutable T] = alloc_mut[T](n_elts);
    let i: uint = n_elts;
    while i > 0u { i -= 1u; v += [mutable t]; }
    ret v;
}

fn buf[T](v: array[T]) -> vbuf { ret rustrt::vec_buf[T](v, 0u); }

fn len[T](v: array[T]) -> uint { ret rustrt::vec_len[T](v); }

fn len_set[T](v: array[T], n: uint) { rustrt::vec_len_set[T](v, n); }

fn buf_off[T](v: array[T], offset: uint) -> vbuf {
    assert (offset < len[T](v));
    ret rustrt::vec_buf[T](v, offset);
}

fn print_debug_info[T](v: array[T]) { rustrt::vec_print_debug_info[T](v); }

// FIXME: typestate precondition (list is non-empty)
// Returns the last element of v.
fn last[@T](v: array[T]) -> option::t[T] {
    let l = len[T](v);
    if l == 0u { ret none[T]; }
    ret some[T](v.(l - 1u));
}


// Returns elements from [start..end) from v.
fn slice[@T](v: array[T], start: uint, end: uint) -> vec[T] {
    assert (start <= end);
    assert (end <= len[T](v));
    let result = alloc[T](end - start);
    let i: uint = start;
    while i < end { result += [v.(i)]; i += 1u; }
    ret result;
}


// FIXME: Should go away eventually.
fn slice_mut[@T](v: array[T], start: uint, end: uint) -> vec[mutable T] {
    assert (start <= end);
    assert (end <= len[T](v));
    let result = alloc_mut[T](end - start);
    let i: uint = start;
    while i < end { result += [mutable v.(i)]; i += 1u; }
    ret result;
}

fn shift[@T](v: &mutable array[T]) -> T {
    let ln = len[T](v);
    assert (ln > 0u);
    let e = v.(0);
    v = slice[T](v, 1u, ln);
    ret e;
}

fn pop[@T](v: &mutable array[T]) -> T {
    let ln = len[T](v);
    assert (ln > 0u);
    ln -= 1u;
    let e = v.(ln);
    v = slice[T](v, 0u, ln);
    ret e;
}

fn top[@T](v: &array[T]) -> T {
    let ln = len[T](v);
    assert (ln > 0u);
    ret v.(ln - 1u);
}

fn push[@T](v: &mutable array[T], t: &T) { v += [t]; }

fn unshift[@T](v: &mutable array[T], t: &T) {
    let rs = alloc[T](len[T](v) + 1u);
    rs += [t];
    rs += v;
    v = rs;
}

fn grow[@T](v: &mutable array[T], n: uint, initval: &T) {
    let i: uint = n;
    while i > 0u { i -= 1u; v += [initval]; }
}

fn grow_set[@T](v: &mutable vec[mutable T], index: uint, initval: &T,
               val: &T) {
    let length = vec::len(v);
    if index >= length { grow(v, index - length + 1u, initval); }
    v.(index) = val;
}

fn grow_init_fn[@T](v: &mutable array[T], n: uint, init_fn: fn() -> T ) {
    let i: uint = n;
    while i > 0u { i -= 1u; v += [init_fn()]; }
}

fn grow_init_fn_set[@T](v: &mutable array[T], index: uint, init_fn: fn() -> T,
                       val: &T) {
    let length = vec::len(v);
    if index >= length { grow_init_fn(v, index - length + 1u, init_fn); }
    v.(index) = val;
}

fn map[@T, @U](f: &fn(&T) -> U , v: &vec[T]) -> vec[U] {
    let rs: vec[U] = alloc[U](len[T](v));
    for ve: T  in v { rs += [f(ve)]; }
    ret rs;
}

fn filter_map[@T, @U](f: &fn(&T) -> option::t[U] , v: &vec[T]) -> vec[U] {
    let rs: vec[U] = [];
    for ve: T  in v { alt f(ve) { some(elt) { rs += [elt]; } none. { } } }
    ret rs;
}

fn map2[@T, @U, @V](f: &operator2[T, U, V], v0: &vec[T], v1: &vec[U])
    -> vec[V] {
    let v0_len = len[T](v0);
    if v0_len != len[U](v1) { fail; }
    let u: vec[V] = alloc[V](v0_len);
    let i = 0u;
    while i < v0_len { u += [f({ v0.(i) }, { v1.(i) })]; i += 1u; }
    ret u;
}

fn find[@T](f: fn(&T) -> bool , v: &vec[T]) -> option::t[T] {
    for elt: T  in v { if f(elt) { ret some[T](elt); } }
    ret none[T];
}

fn position[@T](x: &T, v: &array[T]) -> option::t[uint] {
    let i: uint = 0u;
    while i < len(v) { if x == v.(i) { ret some[uint](i); } i += 1u; }
    ret none[uint];
}

fn position_pred[T](f: fn(&T) -> bool , v: &vec[T]) -> option::t[uint] {
    let i: uint = 0u;
    while i < len(v) { if f(v.(i)) { ret some[uint](i); } i += 1u; }
    ret none[uint];
}

fn member[T](x: &T, v: &array[T]) -> bool {
    for elt: T in v { if x == elt { ret true; } }
    ret false;
}

fn count[T](x: &T, v: &array[T]) -> uint {
    let cnt = 0u;
    for elt: T  in v { if x == elt { cnt += 1u; } }
    ret cnt;
}

fn foldl[@T, @U](p: fn(&U, &T) -> U , z: &U, v: &vec[T]) -> U {
    let sz = len[T](v);
    if sz == 0u {
        ret z;
    } else {
        let rest = slice[T](v, 1u, sz);
        ret p(foldl[T, U](p, z, rest), v.(0));
    }
}

fn unzip[@T, @U](v: &vec[{_0: T, _1: U}]) -> {_0: vec[T], _1: vec[U]} {
    let sz = len(v);
    if sz == 0u {
        ret {_0: alloc[T](0u), _1: alloc[U](0u)};
    } else {
        let rest = slice(v, 1u, sz);
        let tl = unzip[T, U](rest);
        let a = [v.(0)._0];
        let b = [v.(0)._1];
        ret {_0: a + tl._0, _1: b + tl._1};
    }
}


// FIXME make the lengths being equal a constraint
fn zip[@T, @U](v: &vec[T], u: &vec[U]) -> vec[{_0: T, _1: U}] {
    let sz = len(v);
    assert (sz == len(u));
    if sz == 0u {
        ret alloc(0u);
    } else {
        let rest = zip(slice(v, 1u, sz), slice(u, 1u, sz));
        vec::push(rest, {_0: v.(0), _1: u.(0)});
        ret rest;
    }
}

fn or(v: &vec[bool]) -> bool {
    let f = orb;
    ret vec::foldl[bool, bool](f, false, v);
}

fn any[T](f: &fn(&T) -> bool , v: &vec[T]) -> bool {
    for t: T  in v { if f(t) { ret true; } }
    ret false;
}
fn all[T](f: &fn(&T) -> bool , v: &vec[T]) -> bool {
    for t: T  in v { if !f(t) { ret false; } }
    ret true;
}

fn clone[@T](v: &vec[T]) -> vec[T] { ret slice[T](v, 0u, len[T](v)); }

fn plus_option[@T](v: &mutable vec[T], o: &option::t[T]) {
    alt o { none. { } some(x) { v += [x]; } }
}

fn cat_options[@T](v: &vec[option::t[T]]) -> vec[T] {
    let rs: vec[T] = [];
    for o: option::t[T]  in v { alt o { none. { } some(t) { rs += [t]; } } }
    ret rs;
}


// TODO: Remove in favor of built-in "freeze" operation when it's implemented.
fn freeze[@T](v: vec[mutable T]) -> vec[T] {
    let result: vec[T] = [];
    for elem: T  in v { result += [elem]; }
    ret result;
}


// Swaps two elements in a vector
fn swap[@T](v: &vec[mutable T], a: uint, b: uint) {
    let t: T = v.(a);
    v.(a) = v.(b);
    v.(b) = t;
}


// In place vector reversal
fn reverse[@T](v: &vec[mutable T]) {
    let i: uint = 0u;
    let ln = len[T](v);
    while i < ln / 2u { swap(v, i, ln - i - 1u); i += 1u; }
}


// Functional vector reversal. Returns a reversed copy of v.
fn reversed[@T](v: vec[T]) -> vec[T] {
    let rs: vec[T] = [];
    let i = len[T](v);
    if i == 0u { ret rs; } else { i -= 1u; }
    while i != 0u { push[T](rs, v.(i)); i -= 1u; }
    push[T](rs, v.(0));
    ret rs;
}


/// Truncates the vector to length `new_len`.
/// FIXME: This relies on a typechecker bug (covariance vs. invariance).
fn truncate[@T](v: &mutable vec[mutable? T], new_len: uint) {
    v = slice[T](v, 0u, new_len);
}
// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
