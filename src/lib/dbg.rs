/**
 * Unsafe debugging functions for inspecting values.
 */

import std._vec;

native "rust" mod rustrt {
  fn debug_tydesc[T]();
  fn debug_opaque[T](&T x);
  fn debug_box[T](@T x);
  fn debug_tag[T](&T x);
  fn debug_obj[T](&T x, uint nmethods);
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

fn debug_obj[T](&T x, uint nmethods) {
  rustrt.debug_obj[T](x, nmethods);
}

fn debug_fn[T](&T x) {
  rustrt.debug_fn[T](x);
}
