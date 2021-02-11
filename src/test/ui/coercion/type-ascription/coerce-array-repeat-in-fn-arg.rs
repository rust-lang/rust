#![feature(type_ascription)]

fn foo(_arg : [&[u32];3]) {}

fn main() {
  let arr = [4,5,6];
  foo([&arr : &[u32]; 3]);
  //~^ ERROR type ascriptions are not
}
