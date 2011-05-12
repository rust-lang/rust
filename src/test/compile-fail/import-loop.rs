// error-pattern:cyclic import

import x;

fn main() {
  auto y = x;
}
