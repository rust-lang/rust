native "rust" mod rustrt {
  fn last_os_error() -> str;
  fn size_of[T]() -> uint;
  fn align_of[T]() -> uint;
  fn refcount[T](@T t) -> uint;
  fn gc();
}

