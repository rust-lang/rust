#![feature(type_alias_impl_trait)]

pub type Ty<A> = impl Sized + 'static;
#[define_opaque(Ty)]
fn defining<A: 'static>(s: A) -> Ty<A> {
    s
    //~^ ERROR: the parameter type `A` may not live long enough
}
pub fn assert_static<A: 'static>() {}

fn test<A>()
where
    Ty<A>: 'static,
{
    assert_static::<A>()
    //~^ ERROR: the parameter type `A` may not live long enough
}

fn main() {}
