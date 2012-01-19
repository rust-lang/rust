// error-pattern:Declaration of none shadows
import option::*;

fn main() {
  let none: int = 42;
  log(debug, none);
}
