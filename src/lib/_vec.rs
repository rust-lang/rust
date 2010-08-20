import vbuf = rustrt.vbuf;
import op = util.operator;

native "rust" mod rustrt {
  type vbuf;

  fn vec_buf[T](vec[T] v, uint offset) -> vbuf;

  fn vec_len[T](vec[T] v) -> uint;
  /**
   * Sometimes we modify the vec internal data via vec_buf and need to update
   * the vec's fill length accordingly.
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
fn slice[T](vec[T] v, int start, int end) -> vec[T] {
  check (0 <= start);
  check (start <= end);
  check (end <= (len[T](v) as int));
  auto result = alloc[T]((end - start) as uint);
  let int i = start;
  while (i < end) {
    result += vec(v.(i));
    i += 1;
  }
  ret result;
}

fn grow[T](&mutable vec[T] v, int n, &T initval) {
  let int i = n;
  while (i > 0) {
    i -= 1;
    v += vec(initval);
  }
}

fn map[T, U](&op[T,U] f, &vec[T] v) -> vec[U] {
  let vec[U] u = alloc[U](len[T](v));
  for (T ve in v) {
    u += vec(f(ve));
  }
  ret u;
}
