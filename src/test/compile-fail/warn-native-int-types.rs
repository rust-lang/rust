//error-pattern:ctypes::c_int or ctypes::long should be used
native mod xx {
  fn strlen(str: *u8) -> uint;
  fn foo(x: int, y: uint);
}

fn main() {
  // let it fail to verify warning message
  fail
}
