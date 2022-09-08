trait Foo {
    fn answer(self);
}

struct NoData<T>;
//~^ ERROR: parameter `T` is never used

impl<T> Foo for T where NoData<T>: Foo {
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
  fn answer(self) {
    let val: EvenLessData<T> = EvenLessData(None);
  }
}

impl<T> Baz for T where AlmostNoData<T>: Bar {
  fn answer(self) {
    let val: NoData<T> = AlmostNoData(None);
  }
}

fn main() {}
