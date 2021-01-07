#![feature(type_ascription)]

// build-pass

fn foo(_arg : [&[u32];3]) {}

fn main() {
  let arr = [4,5,6];
  foo([&arr : &[u32]; 3]);
}
