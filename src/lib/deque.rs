/**
 * A deque, for fun.  Untested as of yet.  Likely buggy.
 */

import std.util;
import std._vec;
import std._int;

type t[T] = obj {
  fn size() -> uint;

  fn add_front(&T t);
  fn add_back(&T t);

  fn pop_front() -> T;
  fn pop_back() -> T;

  fn peek_front() -> T;
  fn peek_back() -> T;
};

fn create[T]() -> t[T] {

  type cell[T] = mutable util.option[T];

  let uint initial_capacity = uint(32); // 2^5

  /**
   * Grow is only called on full elts, so nelts is also len(elts), unlike
   * elsewhere.
   */
  fn grow[T](uint nelts, uint lo, vec[cell[T]] elts) -> vec[cell[T]] {
    check (nelts == _vec.len[cell[T]](elts));

    fn fill[T](uint i, uint nelts, uint lo, &vec[cell[T]] old) -> cell[T] {
      if (i < nelts) {
        ret old.(int((lo + i) % nelts));
      } else {
        ret util.none[T]();
      }
    }

    let uint nalloc = _int.next_power_of_two(nelts + uint(1));
    let _vec.init_op[cell[T]] copy_op = bind fill[T](_, nelts, lo, elts);
    ret _vec.init_fn[cell[T]](copy_op, nalloc);
  }

  /**
   * FIXME (issue #94): We're converting to int every time we index into the
   * vec, but we really want to index with the lo and hi uints that we have
   * around.
   */

  fn get[T](&vec[cell[T]] elts, uint i) -> T {
    alt (elts.(int(i))) {
      case (util.some[T](t)) { ret t; }
      case (_) { fail; }
    }
  }

  obj deque[T](mutable uint nelts,
               mutable uint lo,
               mutable uint hi,
               mutable vec[cell[T]] elts)
  {
    fn size() -> uint { ret nelts; }

    fn add_front(&T t) {
      let uint oldlo = lo;

      if (lo == uint(0)) {
        lo = _vec.len[cell[T]](elts) - uint(1);
      } else {
        lo -= uint(1);
      }

      if (lo == hi) {
        elts = grow[T](nelts, oldlo, elts);
        lo = _vec.len[cell[T]](elts) - uint(1);
        hi = nelts - uint(1);
      }

      elts.(int(lo)) = util.some[T](t);
      nelts += uint(1);
    }

    fn add_back(&T t) {
      hi = (hi + uint(1)) % _vec.len[cell[T]](elts);

      if (lo == hi) {
        elts = grow[T](nelts, lo, elts);
        lo = uint(0);
        hi = nelts;
      }

      elts.(int(hi)) = util.some[T](t);
      nelts += uint(1);
    }

    /**
     * We actually release (turn to none()) the T we're popping so that we
     * don't keep anyone's refcount up unexpectedly.
     */
    fn pop_front() -> T {
      let T t = get[T](elts, lo);
      elts.(int(lo)) = util.none[T]();
      lo = (lo + uint(1)) % _vec.len[cell[T]](elts);
      ret t;
    }

    fn pop_back() -> T {
      let T t = get[T](elts, hi);
      elts.(int(hi)) = util.none[T]();

      if (hi == uint(0)) {
        hi = _vec.len[cell[T]](elts) - uint(1);
      } else {
        hi -= uint(1);
      }

      ret t;
    }

    fn peek_front() -> T {
      ret get[T](elts, lo);
    }

    fn peek_back() -> T {
      ret get[T](elts, hi);
    }
  }

  let vec[cell[T]] v = _vec.init_elt[cell[T]](util.none[T](),
                                              initial_capacity);

  ret deque[T](uint(0), uint(0), uint(0), v);
}
