// error-pattern: unresolved name: a

mod m1 {
  mod a {
  }
}

fn main(vec[str] args) {
  log m1::a;
}
