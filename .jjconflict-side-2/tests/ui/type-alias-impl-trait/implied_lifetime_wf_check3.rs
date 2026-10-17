#![feature(type_alias_impl_trait)]

mod test_lifetime_param {
    pub type Ty<'a> = impl Sized;
    #[define_opaque(Ty)]
    fn defining(a: &str) -> Ty<'_> {
        a
    }
    pub fn assert_static<'a: 'static>() {}
}
fn test_lifetime_param_test<'a>()
where
    test_lifetime_param::Ty<'a>: 'static,
{
    test_lifetime_param::assert_static::<'a>()
    //~^ ERROR: lifetime may not live long enough
}

mod test_higher_kinded_lifetime_param {
    pub type Ty<'a> = impl Sized + 'a;
    #[define_opaque(Ty)]
    fn defining(a: &str) -> Ty<'_> {
        a
    }
    pub fn assert_static<'a: 'static>() {}
}
fn test_higher_kinded_lifetime_param_test<'a>()
where
    for<'b> test_higher_kinded_lifetime_param::Ty<'b>: 'a,
{
    test_higher_kinded_lifetime_param::assert_static::<'a>()
    //~^ ERROR: lifetime may not live long enough
}

mod test_higher_kinded_lifetime_param2 {
    fn assert_static<'a: 'static>() {}
    fn test<'a>() {
        assert_static::<'a>()
        //~^ ERROR: lifetime may not live long enough
    }
}

mod test_type_param {
    pub type Ty<A> = impl Sized;
    #[define_opaque(Ty)]
    fn defining<A>(s: A) -> Ty<A> {
        s
    }
    pub fn assert_static<A: 'static>() {}
}
fn test_type_param_test<A>()
where
    test_type_param::Ty<A>: 'static,
{
    test_type_param::assert_static::<A>()
    //~^ ERROR: parameter type `A` may not live long enough
}

mod test_implied_from_fn_sig {
    pub type Opaque<T: 'static> = impl Sized;
    #[define_opaque(Opaque)]
    fn defining<T: 'static>() -> Opaque<T> {}

    fn assert_static<T: 'static>() {}

    fn test<T>(_: Opaque<T>) {
        assert_static::<T>();
    }
}

fn main() {}
