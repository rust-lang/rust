// -*- rust -*-
// error-pattern:9:2

/* 1
 * 2
 * 3
 */
fn main() {
  %; // parse error on line 9, but is reported on line 6 instead.
}
