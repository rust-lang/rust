#![allow(warnings)]

trait Id<T> {}
trait Lt<'a> {}

impl<'a> Lt<'a> for () {}
impl<T> Id<T> for T {}

fn free_fn_capture_hrtb_in_impl_trait()
    -> Box<dyn for<'a> Id<impl Lt<'a>>>
        //~^ ERROR `impl Trait` cannot capture higher-ranked lifetime from `dyn` type
{
    Box::new(())
}

struct Foo;
impl Foo {
    fn impl_fn_capture_hrtb_in_impl_trait()
        -> Box<dyn for<'a> Id<impl Lt<'a>>>
            //~^ ERROR `impl Trait` cannot capture higher-ranked lifetime from `dyn` type
    {
        Box::new(())
    }
}

fn main() {}
