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
//~^ ERROR lifetime bounds on nested opaque types are not supported yet
fn make_weird_impl<'b>(x: &'b ()) -> impl for<'a> Hrtb<'a, Assoc = impl Send + 'a> {
    &() //~^ ERROR implementation of `Hrtb` is not general enough
    //~| ERROR lifetime bounds on nested opaque types are not supported yet
}
fn make_bad_impl<'b>(x: &'b ()) -> impl for<'a> Hrtb<'a, Assoc = impl Send + 'a> {
    x //~^ ERROR implementation of `Hrtb` is not general enough
    //~| ERROR lifetime bounds on nested opaque types are not supported yet
}

trait Zend<'a>: Send {}

impl<'a, T: Send> Zend<'a> for T {}

fn make_bad_impl_2<'b>(x: &'b ()) -> impl for<'a> Hrtb<'a, Assoc = impl Zend<'a>> {
    x //~^ ERROR implementation of `Hrtb` is not general enough
    //~| ERROR lifetime bounds on nested opaque types are not supported yet
}

fn main() {}
