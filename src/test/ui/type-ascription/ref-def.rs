#![feature(type_ascription)]

struct Foo {
  a : u32,
}

fn main() {
  let foo = Foo { a : 1 };
  let r = &mut foo;

  let ref x = r : &Foo;
    //~^ ERROR: type ascriptions are not

  let ref another_one = (r : &Foo).a;
    //~^ ERROR: type ascriptions are not

  let arr = [4,5,6];
  let arr_ref = &arr;
  let ref again = (arr_ref : &[u32])[1];
    //~^ ERROR: type ascriptions are not

  let ref last_one = &*(arr_ref : &[u32]);
    //~^ ERROR: type ascriptions are not
}
