/**
 * Unsafe debugging functions for inspecting values.
 *
 * Your RUST_LOG environment variable must contain "stdlib" for any debug
 * logging.
 */

import std._vec;

native "rust" mod rustrt {
  fn debug_tydesc[T]();
  fn debug_opaque[T](&T x);
  fn debug_box[T](@T x);
  fn debug_tag[T](&T x);
  fn debug_obj[T](&T x, uint nmethods, uint nbytes);
  fn debug_fn[T](&T x);
}

fn debug_vec[T](vec[T] v) {
  _vec.print_debug_info[T](v);
}

fn debug_tydesc[T]() {
  rustrt.debug_tydesc[T]();
}

fn debug_opaque[T](&T x) {
  rustrt.debug_opaque[T](x);
}

fn debug_box[T](@T x) {
  rustrt.debug_box[T](x);
}

fn debug_tag[T](&T x) {
  rustrt.debug_tag[T](x);
}

/**
 * `nmethods` is the number of methods we expect the object to have.  The
 * runtime will print this many words of the obj vtbl).
 *
 * `nbytes` is the number of bytes of body data we expect the object to have.
 * The runtime will print this many bytes of the obj body.  You probably want
 * this to at least be 4u, since an implicit captured tydesc pointer sits in
 * the front of any obj's data tuple.x
 */
fn debug_obj[T](&T x, uint nmethods, uint nbytes) {
  rustrt.debug_obj[T](x, nmethods, nbytes);
}

fn debug_fn[T](&T x) {
  rustrt.debug_fn[T](x);
}
