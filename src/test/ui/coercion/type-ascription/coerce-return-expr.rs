#![feature(type_ascription)]

fn foo<'a>(arg : (u32, (u32, &'a [u32; 3]))) -> (u32, (u32, &'a [u32])) {
  arg : (u32, (u32, &[u32]))
    //~^ ERROR: mismatched types
}

fn main() {
  let arr = [4,5,6];
  let tup = (3, (9, &arr));
  let result = foo(tup);
}
