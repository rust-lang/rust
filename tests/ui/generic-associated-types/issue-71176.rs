trait Provider {
    type A<'a>;
}

impl Provider for () {
    type A<'a> = ();
}

struct Holder<B> {
  inner: Box<dyn Provider<A = B>>,
  //~^ ERROR: missing generics for associated type
}

fn main() {
    Holder {
        inner: Box::new(()),
    };
}
