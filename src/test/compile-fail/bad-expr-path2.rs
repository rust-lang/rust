// error-pattern: is a module, not a

mod m1 {
  mod a {
  }
}

fn main(vec[str] args) {
  log m1::a;
}
