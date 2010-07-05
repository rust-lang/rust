export rustrt;

native "rust" mod rustrt {

  // Explicitly re-export native stuff we want to be made
  // available outside this crate. Otherwise it's
  // visible-in-crate, but not re-exported.

  export last_os_error;
  export size_of;
  export align_of;
  export refcount;
  export gc;

  fn last_os_error() -> str;
  fn size_of[T]() -> uint;
  fn align_of[T]() -> uint;
  fn refcount[T](@T t) -> uint;
  fn gc();
  fn unsupervise();
}

