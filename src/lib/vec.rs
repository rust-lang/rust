import option::none;
import option::some;
import util::orb;

type vbuf = rustrt::vbuf;

type operator2[T,U,V] = fn(&T, &U) -> V;

type array[T] = vec[mutable? T];

native "rust" mod rustrt {
    type vbuf;

    fn vec_buf[T](vec[T] v, uint offset) -> vbuf;

    fn vec_len[T](vec[T] v) -> uint;
    /**
     * Sometimes we modify the vec internal data via vec_buf and need to
     * update the vec's fill length accordingly.
     */
    fn vec_len_set[T](vec[T] v, uint n);

    /**
     * The T in vec_alloc[T, U] is the type of the vec to allocate.  The
     * U is the type of an element in the vec.  So to allocate a vec[U] we
     * want to invoke this as vec_alloc[vec[U], U].
     */
    fn vec_alloc[T, U](uint n_elts) -> vec[U];
    fn vec_alloc_mut[T, U](uint n_elts) -> vec[mutable U];

    fn refcount[T](vec[T] v) -> uint;

    fn vec_print_debug_info[T](vec[T] v);

    fn vec_from_vbuf[T](vbuf v, uint n_elts) -> vec[T];

    fn unsafe_vec_to_mut[T](vec[T] v) -> vec[mutable T];
}

fn alloc[T](uint n_elts) -> vec[T] {
    ret rustrt::vec_alloc[vec[T], T](n_elts);
}

fn alloc_mut[T](uint n_elts) -> vec[mutable T] {
    ret rustrt::vec_alloc_mut[vec[mutable T], T](n_elts);
}

fn refcount[T](array[T] v) -> uint {
    auto r = rustrt::refcount[T](v);
    if (r == dbg::const_refcount) {
        ret r;
    } else {
        // -2 because calling this function and the native function both
        // incremented the refcount.
        ret  r - 2u;
    }
}

fn vec_from_vbuf[T](vbuf v, uint n_elts) -> vec[T] {
    ret rustrt::vec_from_vbuf[T](v, n_elts);
}

// FIXME: Remove me; this is a botch to get around rustboot's bad typechecker.
fn empty[T]() -> vec[T] {
    ret alloc[T](0u);
}

// FIXME: Remove me; this is a botch to get around rustboot's bad typechecker.
fn empty_mut[T]() -> vec[mutable T] {
    ret alloc_mut[T](0u);
}

type init_op[T] = fn(uint i) -> T;

fn init_fn[T](&init_op[T] op, uint n_elts) -> vec[T] {
    let vec[T] v = alloc[T](n_elts);
    let uint i = 0u;
    while (i < n_elts) {
        v += [op(i)];
        i += 1u;
    }
    ret v;
}

fn init_fn_mut[T](&init_op[T] op, uint n_elts) -> vec[mutable T] {
    let vec[mutable T] v = alloc_mut[T](n_elts);
    let uint i = 0u;
    while (i < n_elts) {
        v += [mutable op(i)];
        i += 1u;
    }
    ret v;
}

fn init_elt[T](&T t, uint n_elts) -> vec[T] {
    /**
     * FIXME (issue #81): should be:
     *
     * fn elt_op[T](&T x, uint i) -> T { ret x; }
     * let init_op[T] inner = bind elt_op[T](t, _);
     * ret init_fn[T](inner, n_elts);
     */
    let vec[T] v = alloc[T](n_elts);
    let uint i = n_elts;
    while (i > 0u) {
        i -= 1u;
        v += [t];
    }
    ret v;
}

fn init_elt_mut[T](&T t, uint n_elts) -> vec[mutable T] {
    let vec[mutable T] v = alloc_mut[T](n_elts);
    let uint i = n_elts;
    while (i > 0u) {
        i -= 1u;
        v += [mutable t];
    }
    ret v;
}

fn buf[T](array[T] v) -> vbuf {
    ret rustrt::vec_buf[T](v, 0u);
}

fn len[T](array[T] v) -> uint {
    ret rustrt::vec_len[T](v);
}

fn len_set[T](array[T] v, uint n) {
    rustrt::vec_len_set[T](v, n);
}

fn buf_off[T](array[T] v, uint offset) -> vbuf {
     assert (offset < len[T](v));
    ret rustrt::vec_buf[T](v, offset);
}

fn print_debug_info[T](array[T] v) {
    rustrt::vec_print_debug_info[T](v);
}

// Returns the last element of v.
fn last[T](array[T] v) -> option::t[T] {
    auto l = len[T](v);
    if (l == 0u) {
        ret none[T];
    }
    ret some[T](v.(l - 1u));
}

// Returns elements from [start..end) from v.

fn slice[T](array[T] v, uint start, uint end) -> vec[T] {
    assert (start <= end);
    assert (end <= len[T](v));
    auto result = alloc[T](end - start);
    let uint i = start;
    while (i < end) {
        result += [v.(i)];
        i += 1u;
    }
    ret result;
}

// FIXME: Should go away eventually.
fn slice_mut[T](array[T] v, uint start, uint end) -> vec[mutable T] {
    assert (start <= end);
    assert (end <= len[T](v));
    auto result = alloc_mut[T](end - start);
    let uint i = start;
    while (i < end) {
        result += [mutable v.(i)];
        i += 1u;
    }
    ret result;
}

fn shift[T](&mutable array[T] v) -> T {
    auto ln = len[T](v);
    assert (ln > 0u);
    auto e = v.(0);
    v = slice[T](v, 1u, ln);
    ret e;
}

fn pop[T](&mutable array[T] v) -> T {
    auto ln = len[T](v);
    assert (ln > 0u);
    ln -= 1u;
    auto e = v.(ln);
    v = slice[T](v, 0u, ln);
    ret e;
}

fn top[T](&array[T] v) -> T {
    auto ln = len[T](v);
    assert (ln > 0u);
    ret v.(ln-1u);
}

fn push[T](&mutable array[T] v, &T t) {
    v += [t];
}

fn unshift[T](&mutable array[T] v, &T t) {
    auto res = alloc[T](len[T](v) + 1u);
    res += [t];
    res += v;
    v = res;
}

fn grow[T](&mutable array[T] v, uint n, &T initval) {
    let uint i = n;
    while (i > 0u) {
        i -= 1u;
        v += [initval];
    }
}

fn grow_set[T](&mutable vec[mutable T] v, uint index, &T initval, &T val) {
    auto length = vec::len(v);
    if (index >= length) {
        grow(v, index - length + 1u, initval);
    }
    v.(index) = val;
}

fn grow_init_fn[T](&array[T] v, uint n, fn()->T init_fn) {
    let uint i = n;
    while (i > 0u) {
        i -= 1u;
        v += [init_fn()];
    }
}

fn grow_init_fn_set[T](&array[T] v, uint index, fn()->T init_fn, &T val) {
    auto length = vec::len(v);
    if (index >= length) { grow_init_fn(v, index - length + 1u, init_fn); }
    v.(index) = val;
}


fn map[T, U](&fn(&T) -> U f, &vec[T] v) -> vec[U] {
    let vec[U] res = alloc[U](len[T](v));
    for (T ve in v) {
        res += [f(ve)];
    }
    ret res;
}

fn filter_map[T, U](&fn(&T) -> option::t[U] f, &vec[T] v) -> vec[U] {
    let vec[U] res = [];
    for(T ve in v) {
        alt(f(ve)) {
            case (some(?elt)) { res += [elt]; }
            case (none) {}
        }
    }
    ret res;
}

fn map2[T,U,V](&operator2[T,U,V] f, &vec[T] v0, &vec[U] v1) -> vec[V] {
    auto v0_len = len[T](v0);
    if (v0_len != len[U](v1)) {
        fail;
    }

    let vec[V] u = alloc[V](v0_len);
    auto i = 0u;
    while (i < v0_len) {
        u += [f({v0.(i)}, {v1.(i)})];
        i += 1u;
    }

    ret u;
}

fn find[T](fn (&T) -> bool f, &vec[T] v) -> option::t[T] {
    for (T elt in v) {
        if (f(elt)) {
            ret some[T](elt);
        }
    }

    ret none[T];
}

fn member[T](&T x, &array[T] v) -> bool {
    for (T elt in v) {
        if (x == elt) { ret true; }
    }
    ret false;
}

fn foldl[T, U](fn (&U, &T) -> U p, &U z, &vec[T] v) -> U {
    auto sz = len[T](v);

    if (sz == 0u) {
        ret z;
    }
    else {
        auto rest = slice[T](v, 1u, sz);

        ret (p(foldl[T,U](p, z, rest), v.(0)));
    }
}

fn unzip[T, U](&vec[tup(T, U)] v) -> tup(vec[T], vec[U]) {
    auto sz = len[tup(T, U)](v);

    if (sz == 0u) {
        ret tup(alloc[T](0u), alloc[U](0u));
    }
    else {
        auto rest = slice[tup(T, U)](v, 1u, sz);
        auto tl   = unzip[T, U](rest);
        auto a    = [v.(0)._0];
        auto b    = [v.(0)._1];
        ret tup(a + tl._0, b + tl._1);
    }
}

// FIXME make the lengths being equal a constraint
fn zip[T, U](&vec[T] v, &vec[U] u) -> vec[tup(T, U)] {
    auto sz = len[T](v);
    assert (sz == len[U](u));

    if (sz == 0u) {
        ret alloc[tup(T, U)](0u);
    }
    else {
        auto rest = zip[T, U](slice[T](v, 1u, sz), slice[U](u, 1u, sz));
        vec::push(rest, tup(v.(0), u.(0)));
        ret rest;
    }
}

fn or(&vec[bool] v) -> bool {
    auto f = orb;
    ret vec::foldl[bool, bool](f, false, v);
}

fn clone[T](&vec[T] v) -> vec[T] {
    ret slice[T](v, 0u, len[T](v));
}

fn plus_option[T](&vec[T] v, &option::t[T] o) -> () {
    alt (o) {
        case (none) {}
        case (some(?x)) { v += [x]; }
    }
}

fn cat_options[T](&vec[option::t[T]] v) -> vec[T] {
    let vec[T] res = [];

    for (option::t[T] o in v) {
        alt (o) {
            case (none) { }
            case (some(?t)) {
                res += [t];
            }
        }
    }

    ret res;
}

// TODO: Remove in favor of built-in "freeze" operation when it's implemented.
fn freeze[T](vec[mutable T] v) -> vec[T] {
    let vec[T] result = [];
    for (T elem in v) {
        result += [elem];
    }
    ret result;
}

// Swaps two elements in a vector
fn swap[T](&vec[T] v, uint a, uint b) {
    let T t = v.(a);
    v.(a) = v.(b);
    v.(b) = t;
}

// In place vector reversal
fn reverse[T](&vec[T] v) -> () {
    let uint i = 0u;
    auto ln = len[T](v);

    while(i < ln / 2u) {
        swap(v, i, ln - i - 1u);
        i += 1u;
    }
}

// Functional vector reversal. Returns a reversed copy of v.
fn reversed[T](vec[T] v) -> vec[T] {
    let vec[T] res = [];

    auto i = len[T](v);
    if (i == 0u) {
        ret res;
    }
    else {
        i -= 1u;
    }

    while(i != 0u) {
        push[T](res, v.(i));
        i -= 1u;
    }
    push[T](res, v.(0));

    ret res;
}

/// Truncates the vector to length `new_len`.
/// FIXME: This relies on a typechecker bug (covariance vs. invariance).
fn truncate[T](&mutable vec[mutable? T] v, uint new_len) {
    v = slice[T](v, 0u, new_len);
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C .. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
