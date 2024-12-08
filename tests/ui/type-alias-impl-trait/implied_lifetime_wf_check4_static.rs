#![feature(type_alias_impl_trait)]

mod test_type_param_static {
    pub type Ty<A> = impl Sized + 'static;
    fn defining<A: 'static>(s: A) -> Ty<A> {
        s
        //~^ ERROR: the parameter type `A` may not live long enough
    }
    pub fn assert_static<A: 'static>() {}
}
use test_type_param_static::*;

fn test<A>()
where
    Ty<A>: 'static,
{
    assert_static::<A>()
    //~^ ERROR: the parameter type `A` may not live long enough
}

fn main() {}
