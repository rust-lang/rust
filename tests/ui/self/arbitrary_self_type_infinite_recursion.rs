#![feature(arbitrary_self_types)]

struct MySmartPtr<T>(T);

impl<T> core::ops::Receiver for MySmartPtr<T> {
  type Target = MySmartPtr<T>;
}

struct Content;

impl Content {
  fn method(self: MySmartPtr<Self>) { // note self type
     //~^ ERROR reached the recursion limit
     //~| ERROR reached the recursion limit
     //~| ERROR invalid `self` parameter type
  }
}

fn main() {
  let p = MySmartPtr(Content);
  p.method();
  //~^ ERROR reached the recursion limit
  //~| ERROR no method named `method`
}
