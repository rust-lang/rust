// build-pass

#![feature(type_ascription)]

union Foo<'a> {
    f1: (&'a u32, (u32, &'a [u32])),
    _f2: u32,
}

fn main() {
  let arr = [4,5,6];
  let x = &mut 26;
  let _ = Foo { f1: (x : &u32, (5, &arr : &[u32])) };
}
