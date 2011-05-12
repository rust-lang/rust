// xfail-boot
// error-pattern: unresolved name: a

mod m1 {
}

fn main(vec[str] args) {
  log m1::a;
}
