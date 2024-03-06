//@ run-pass
use std::marker::PhantomData;

#[derive(Debug)]
struct LifetimeStruct<'a>(PhantomData<&'a ()>);

fn main() {
    takes_hrtb_closure(|lts| println!("{:?}", lts));
}

fn takes_hrtb_closure<F: for<'a>FnMut(LifetimeStruct<'a>)>(mut f: F) {
    f(LifetimeStruct(PhantomData));
}
