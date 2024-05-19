//@ run-pass
#![allow(dead_code)]

pub fn main() {
  trait Text {
    fn to_string(&self) -> String;
  }

  fn to_string(t: Box<dyn Text>) {
    println!("{}", (*t).to_string());
  }

}
