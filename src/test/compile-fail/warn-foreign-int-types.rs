//error-pattern:libc::c_int or libc::c_long should be used
extern mod xx {
    #[legacy_exports];
  fn strlen(str: *u8) -> uint;
  fn foo(x: int, y: uint);
}

fn main() {
  // let it fail to verify warning message
  fail
}
