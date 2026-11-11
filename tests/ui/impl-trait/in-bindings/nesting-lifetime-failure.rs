#![feature(impl_trait_in_bindings)]

trait Static {}
impl<T: 'static> Static for T {}

fn main() {
    let local = 0;
    let _: impl IntoIterator<Item = impl Static> = [&local];
    //~^ ERROR `local` does not live long enough
}
