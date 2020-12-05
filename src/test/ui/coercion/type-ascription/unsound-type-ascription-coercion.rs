#![feature(type_ascription)]

fn f() {
  let mut x: &'static u32 = &22;
  let y = &44;
  foo(x, y);
}

fn foo<'a, 'b>(arg1 : &'a u32, arg2 : &'b u32) {
  let p = &mut (arg1 : &'b u32);
    //~^ ERROR: lifetime mismatch
  *p = arg2;
}

fn main() {
  f();
}
