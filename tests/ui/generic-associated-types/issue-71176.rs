trait Provider {
    type A<'a>;
}

impl Provider for () {
    type A<'a> = ();
}

struct Holder<B> {
  inner: Box<dyn Provider<A = B>>,
  //~^ ERROR: missing generics for associated type
  //~| ERROR: missing generics for associated type
  //~| ERROR: missing generics for associated type
  //~| ERROR: the trait `Provider` is not dyn compatible
}

fn main() {
    Holder {
        inner: Box::new(()),
        //~^ ERROR: the trait `Provider` is not dyn compatible
    };
}
