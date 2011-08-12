
import option::none;
import option::some;
import util::orb;

export vbuf;

type vbuf = rustrt::vbuf;

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

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
