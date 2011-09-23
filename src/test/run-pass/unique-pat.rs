fn simple() {
    alt ~true {
      ~true { }
      _ { fail; }
    }
}

fn main() {
    simple();
}