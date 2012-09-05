// error-pattern:declaration of `None` shadows
use option::*;

fn main() {
  let None: int = 42;
  log(debug, None);
}
