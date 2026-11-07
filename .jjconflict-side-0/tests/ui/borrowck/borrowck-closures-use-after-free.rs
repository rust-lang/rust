// Tests that a closure which mutates a local variable
// cannot also be supplied a borrowed version of that
// variable's contents. Issue #11192.

struct Foo {
  x: isize
}

impl Drop for Foo {
  fn drop(&mut self) {
    println!("drop {}", self.x);
  }
}



fn main() {
  let mut ptr: Box<_> = Box::new(Foo { x: 0 });
  let mut test = |foo: &Foo| {
    ptr = Box::new(Foo { x: ptr.x + 1 });
  };
  test(&*ptr); //~ ERROR cannot borrow `*ptr`
}
