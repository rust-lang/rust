#[legacy_exports];
unsafe fn f(xs: ~[int]) {
  xs.map(|_x| { unsafe fn q() { fail; } });
}