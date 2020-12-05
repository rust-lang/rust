// build-pass

#![feature(type_ascription)]

enum Foo<'a> {
  A((u32, &'a [u32])),
  B((u32, &'a [u32; 4])),
}

fn main() {
  let arr = [4,5,6];
  let temp = Foo::A((10, &arr : &[u32]));
}
