// Parse error on line X, but is reported on line Y instead.

/* 1
 * 2
 * 3
 */
fn main() {
  %; //~ ERROR expected expression, found `%`
}
