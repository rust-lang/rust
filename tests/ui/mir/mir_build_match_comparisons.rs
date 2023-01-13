// run-pass
#![allow(dead_code)]
fn test1(x: i8) -> i32 {
  match x {
    1..=10 => 0,
    _ => 1,
  }
}

const U: Option<i8> = Some(10);
const S: &'static str = "hello";

fn test2(x: i8) -> i32 {
  match Some(x) {
    U => 0,
    _ => 1,
  }
}

fn test3(x: &'static str) -> i32 {
  match x {
    S => 0,
    _ => 1,
  }
}

enum Opt<T> {
    Some { v: T },
    None
}

fn test4(x: u64) -> i32 {
  let opt = Opt::Some{ v: x };
  match opt {
    Opt::Some { v: 10 } => 0,
    _ => 1,
  }
}


fn main() {
  assert_eq!(test1(0), 1);
  assert_eq!(test1(1), 0);
  assert_eq!(test1(2), 0);
  assert_eq!(test1(5), 0);
  assert_eq!(test1(9), 0);
  assert_eq!(test1(10), 0);
  assert_eq!(test1(11), 1);
  assert_eq!(test1(20), 1);
  assert_eq!(test2(10), 0);
  assert_eq!(test2(0), 1);
  assert_eq!(test2(20), 1);
  assert_eq!(test3("hello"), 0);
  assert_eq!(test3(""), 1);
  assert_eq!(test3("world"), 1);
  assert_eq!(test4(10), 0);
  assert_eq!(test4(0), 1);
  assert_eq!(test4(20), 1);
}
