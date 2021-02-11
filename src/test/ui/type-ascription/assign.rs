#![feature(type_ascription)]

fn main() {
  let arr1 = [4,5,6];
  let arr2 = [2,2,2];
  let mut x = &arr1;
  x : &[u32] = &arr2;
    //~^ ERROR type ascriptions are not allowed
    //~| ERROR invalid left-hand side of assignment
}
