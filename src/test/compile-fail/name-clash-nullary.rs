// error-pattern:declaration of `None` shadows
import option::*;

fn main() {
  let None: int = 42;
  log(debug, None);
}
