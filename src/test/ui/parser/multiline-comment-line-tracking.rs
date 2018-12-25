// compile-flags: -Z parse-only
// error-pattern:9:3

/* 1
 * 2
 * 3
 */
fn main() {
  %; // parse error on line 19, but is reported on line 6 instead.
}
