#![allow(warnings)]

trait Id<T> {}
trait Lt<'a> {}

impl<'a> Lt<'a> for () {}
impl<T> Id<T> for T {}

fn free_fn_capture_hrtb_in_impl_trait()
    -> Box<for<'a> Id<impl Lt<'a>>>
//~^ ERROR `impl Trait` can only capture lifetimes bound at the fn or impl level [E0657]
//~| ERROR lifetime bounds on nested opaque types are not supported yet
{
    Box::new(())
}

struct Foo;
impl Foo {
    fn impl_fn_capture_hrtb_in_impl_trait()
        -> Box<for<'a> Id<impl Lt<'a>>>
    //~^ ERROR `impl Trait` can only capture lifetimes bound at the fn or impl level
    //~| ERROR lifetime bounds on nested opaque types are not supported yet
    {
        Box::new(())
    }
}

fn main() {}
