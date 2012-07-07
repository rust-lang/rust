// error-pattern:variant e doesn't belong to enum floop
import bad::*;

mod bad {

  export floop::{a, e};

  enum floop {a, b, c}
  enum bloop {d, e, f}

}

fn main() {
}