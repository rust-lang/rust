// error-pattern:18:3

/* 1
 * 2
 * 3
 */
fn main() {
  %; // parse error on line 18, but is reported on line 6 instead.
}
