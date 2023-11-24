// check-pass

trait Call {
  type Out;
  fn call(self) -> Self::Out;
}

impl<F: FnOnce() -> T, T> Call for F {
  type Out = T;
  fn call(self) -> T { (self)() }
}

pub struct Foo {
    bar: u8
}

fn diverge() -> ! { todo!() }

#[allow(unused_variables)]
fn main() {
    let Foo { ref bar } = diverge.call();
}
