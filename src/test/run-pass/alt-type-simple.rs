
// xfail-test

fn altsimple(any x) {
  match type (f) {
    case (int i) { print("int"); }
    case (str s) { print("str"); }
  }
}

pub fn main() {
  altsimple(5);
  altsimple("asdfasdfsDF");
}
