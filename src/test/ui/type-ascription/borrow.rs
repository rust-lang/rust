#![feature(type_ascription)]

struct Foo {
  a : u32,
}

fn main() {
  let foo = Foo { a : 1 };
  let r = &mut foo;

  let x = &(r : &Foo);
    //~^ ERROR: type ascriptions are not

  let another_one = &(r : &Foo).a;
    //~^ ERROR: type ascriptions are not

  let arr = [4,5,6];
  let arr_ref = &arr;
  let ref last_one = &(arr_ref : &[u32])[1];
    //~^ ERROR: type ascriptions are not
}
