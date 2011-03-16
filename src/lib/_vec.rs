import vbuf = rustrt.vbuf;

type operator2[T,U,V] = fn(&T, &U) -> V;

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

    fn refcount[T](vec[T] v) -> uint;

    fn vec_print_debug_info[T](vec[T] v);
}

fn alloc[T](uint n_elts) -> vec[T] {
    ret rustrt.vec_alloc[vec[T], T](n_elts);
}

fn refcount[T](vec[T] v) -> uint {
    auto r = rustrt.refcount[T](v);
    if (r == dbg.const_refcount) {
        ret r;
    } else {
        // -1 because calling this function incremented the refcount.
        ret  r - 1u;
    }
}

type init_op[T] = fn(uint i) -> T;

fn init_fn[T](&init_op[T] op, uint n_elts) -> vec[T] {
    let vec[T] v = alloc[T](n_elts);
    let uint i = 0u;
    while (i < n_elts) {
        v += vec(op(i));
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
        v += vec(t);
    }
    ret v;
}

fn buf[T](vec[T] v) -> vbuf {
    ret rustrt.vec_buf[T](v, 0u);
}

fn len[T](vec[T] v) -> uint {
    ret rustrt.vec_len[T](v);
}

fn len_set[T](vec[T] v, uint n) {
    rustrt.vec_len_set[T](v, n);
}

fn buf_off[T](vec[T] v, uint offset) -> vbuf {
    check (offset < len[T](v));
    ret rustrt.vec_buf[T](v, offset);
}

fn print_debug_info[T](vec[T] v) {
    rustrt.vec_print_debug_info[T](v);
}

// Returns elements from [start..end) from v.
fn slice[T](vec[T] v, uint start, uint end) -> vec[T] {
    check (start <= end);
    check (end <= len[T](v));
    auto result = alloc[T](end - start);
    let uint i = start;
    while (i < end) {
        result += vec(v.(i));
        i += 1u;
    }
    ret result;
}

fn shift[T](&mutable vec[T] v) -> T {
    auto ln = len[T](v);
    check(ln > 0u);
    auto e = v.(0);
    v = slice[T](v, 1u, ln);
    ret e;
}

fn pop[T](&mutable vec[T] v) -> T {
    auto ln = len[T](v);
    check(ln > 0u);
    ln -= 1u;
    auto e = v.(ln);
    v = slice[T](v, 0u, ln);
    ret e;
}

fn push[T](&mutable vec[T] v, &T t) {
    v += vec(t);
}

fn unshift[T](&mutable vec[T] v, &T t) {
    auto res = alloc[T](len[T](v) + 1u);
    res += vec(t);
    res += v;
    v = res;
}

fn grow[T](&mutable vec[T] v, int n, &T initval) {
    let int i = n;
    while (i > 0) {
        i -= 1;
        v += vec(initval);
    }
}

fn map[T, U](&option.operator[T,U] f, &vec[T] v) -> vec[U] {
    let vec[U] u = alloc[U](len[T](v));
    for (T ve in v) {
        u += vec(f(ve));
    }
    ret u;
}

fn map2[T,U,V](&operator2[T,U,V] f, &vec[T] v0, &vec[U] v1) -> vec[V] {
    auto v0_len = len[T](v0);
    if (v0_len != len[U](v1)) {
        fail;
    }

    let vec[V] u = alloc[V](v0_len);
    auto i = 0u;
    while (i < v0_len) {
        u += vec(f(v0.(i), v1.(i)));
        i += 1u;
    }

    ret u;
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C .. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
