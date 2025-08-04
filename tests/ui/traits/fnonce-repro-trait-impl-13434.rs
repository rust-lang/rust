//! Regression test for https://github.com/rust-lang/rust/issues/13434

//@ run-pass
#[derive(Debug)]
struct MyStruct;

trait Repro {
  fn repro(self, s: MyStruct) -> String;
}

impl<F> Repro for F where F: FnOnce(MyStruct) -> String {
  fn repro(self, s: MyStruct) -> String {
    self(s)
  }
}

fn do_stuff<R: Repro>(r: R) -> String {
  r.repro(MyStruct)
}

pub fn main() {
  assert_eq!("MyStruct".to_string(), do_stuff(|s: MyStruct| format!("{:?}", s)));
}
