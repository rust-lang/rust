// error-pattern: can't refer to a module as a first-class value

mod m1 {
  mod a {
  }
}

fn main(vec[str] args) {
  log m1.a;
}
