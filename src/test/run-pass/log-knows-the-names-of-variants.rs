#![allow(non_camel_case_types)]
#![allow(dead_code)]
#[derive(Debug)]
enum foo {
  a(usize),
  b(String),
  c,
}

#[derive(Debug)]
enum bar {
  d, e, f
}

pub fn main() {
    assert_eq!("a(22)".to_string(), format!("{:?}", foo::a(22)));
    assert_eq!("c".to_string(), format!("{:?}", foo::c));
    assert_eq!("d".to_string(), format!("{:?}", bar::d));
}
