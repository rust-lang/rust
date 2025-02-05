#![feature(arbitrary_self_types)]

struct MySmartPtr<T>(T);

impl<T> core::ops::Receiver for MySmartPtr<T> {
  type Target = MySmartPtr<T>;
}

struct Content;

impl Content {
  fn method(self: MySmartPtr<Self>) { // note self type
     //~^ reached the recursion limit
     //~| reached the recursion limit
     //~| invalid `self` parameter type
  }
}

fn main() {
  let p = MySmartPtr(Content);
  p.method();
  //~^ reached the recursion limit
  //~| no method named `method`
}
