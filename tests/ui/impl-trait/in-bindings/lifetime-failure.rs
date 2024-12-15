#![feature(impl_trait_in_bindings)]

trait Static: 'static {}
impl<T: 'static> Static for T {}

struct W<T>(T);

fn main() {
    let local = 0;
    let _: W<impl Static> = W(&local);
    //~^ ERROR `local` does not live long enough
}
