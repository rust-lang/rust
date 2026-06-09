// https://github.com/rust-lang/rust/issues/20413
trait Foo {
    fn answer(self);
}

struct NoData<T>;
//~^ ERROR: parameter `T` is never used

impl<T> Foo for T where NoData<T>: Foo {
  //~^ ERROR: overflow evaluating the requirement
  fn answer(self) {
    let val: NoData<T> = NoData;
  }
}

trait Bar {
    fn answer(self);
}

trait Baz {
    fn answer(self);
}

struct AlmostNoData<T>(Option<T>);

struct EvenLessData<T>(Option<T>);

impl<T> Bar for T where EvenLessData<T>: Baz {
//~^ ERROR: overflow evaluating the requirement
  fn answer(self) {
    let val: EvenLessData<T> = EvenLessData(None);
  }
}

impl<T> Baz for T where AlmostNoData<T>: Bar {
//~^ ERROR: overflow evaluating the requirement
  fn answer(self) {
    let val: NoData<T> = AlmostNoData(None);
  }
}

fn main() {}
