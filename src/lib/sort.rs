import _vec.len;
import _vec.slice;

type lteq[T] = fn(&T a, &T b) -> bool;

fn merge_sort[T](lteq[T] le, vec[T] v) -> vec[T] {

  fn merge[T](lteq[T] le, vec[T] a, vec[T] b) -> vec[T] {
    let vec[T] res = vec();
    let uint a_len = len[T](a);
    let uint a_ix = 0u;
    let uint b_len = len[T](b);
    let uint b_ix = 0u;
    while (a_ix < a_len && b_ix < b_len) {
      if (le(a.(a_ix), b.(b_ix))) {
        res += a.(a_ix);
        a_ix += 1u;
      } else {
        res += b.(b_ix);
        b_ix += 1u;
      }
    }
    res += slice[T](a, a_ix, a_len);
    res += slice[T](b, b_ix, b_len);
    ret res;
  }

  let uint v_len = len[T](v);

  if (v_len <= 1u) {
    ret v;
  }

  let uint mid = v_len / 2u;
  let vec[T] a = slice[T](v, 0u, mid);
  let vec[T] b = slice[T](v, mid, v_len);
  ret merge[T](le,
               merge_sort[T](le, a),
               merge_sort[T](le, b));
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C .. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
