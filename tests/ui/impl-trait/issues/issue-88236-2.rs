// this used to cause stack overflows

trait Hrtb<'a> {
    type Assoc;
}

impl<'a> Hrtb<'a> for () {
    type Assoc = ();
}

impl<'a> Hrtb<'a> for &'a () {
    type Assoc = ();
}

fn make_impl() -> impl for<'a> Hrtb<'a, Assoc = impl Send + 'a> {}
//~^ ERROR `impl Trait` cannot capture higher-ranked lifetime from outer `impl Trait`

fn make_weird_impl<'b>(x: &'b ()) -> impl for<'a> Hrtb<'a, Assoc = impl Send + 'a> {
    //~^ ERROR `impl Trait` cannot capture higher-ranked lifetime from outer `impl Trait`
    &()
    //~^ ERROR implementation of `Hrtb` is not general enough
    //~| ERROR implementation of `Hrtb` is not general enough
}

fn make_bad_impl<'b>(x: &'b ()) -> impl for<'a> Hrtb<'a, Assoc = impl Send + 'a> {
    //~^ ERROR `impl Trait` cannot capture higher-ranked lifetime from outer `impl Trait`
    x
    //~^ ERROR implementation of `Hrtb` is not general enough
    //~| ERROR implementation of `Hrtb` is not general enough
    //~| ERROR lifetime may not live long enough
}

fn main() {}
