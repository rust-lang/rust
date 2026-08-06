// Test that we show the correct type parameter that couldn't be inferred and that we don't
// end up stating nonsense like "type parameter `'a`" which we used to do.

trait Trait<'a, T> {
    fn m(self);
}

impl<'a, A> Trait<'a, A> for () {
    fn m(self) {}
}

fn qualified() {
    <() as Trait<'static, _>>::m(());
    //~^ ERROR type annotations needed
    //~| NOTE cannot infer type of the type parameter `T`

}

fn unqualified() {
    Trait::<'static, _>::m(());
    //~^ ERROR type annotations needed
    //~| NOTE cannot infer type of the type parameter `T`
}

fn main() {}
