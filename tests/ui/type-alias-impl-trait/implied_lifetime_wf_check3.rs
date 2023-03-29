#![feature(type_alias_impl_trait)]

#[rustfmt::skip]
mod test_lifetime_param {
    type Ty<'a> = impl Sized + 'a;
    #[defines(Ty<'a>)]
    fn defining<'a>(a: &'a str) -> Ty<'a> { a }
    fn assert_static<'a: 'static>() {}
    fn test<'a>() where Ty<'a>: 'static { assert_static::<'a>() }
    //~^ ERROR: lifetime may not live long enough
}

#[rustfmt::skip]
mod test_higher_kinded_lifetime_param {
    type Ty<'a> = impl Sized + 'a;
    #[defines(Ty<'a>)]
    fn defining<'a>(a: &'a str) -> Ty<'a> { a }
    fn assert_static<'a: 'static>() {}
    fn test<'a>() where for<'b> Ty<'b>: 'a { assert_static::<'a>() }
    //~^ ERROR: lifetime may not live long enough
}

#[rustfmt::skip]
mod test_higher_kinded_lifetime_param2 {
    fn assert_static<'a: 'static>() {}
    fn test<'a>() { assert_static::<'a>() }
    //~^ ERROR: lifetime may not live long enough
}

#[rustfmt::skip]
mod test_type_param {
    type Ty<A> = impl Sized;
    #[defines(Ty<A>)]
    fn defining<A>(s: A) -> Ty<A> { s }
    fn assert_static<A: 'static>() {}
    fn test<A>() where Ty<A>: 'static { assert_static::<A>() }
    //~^ ERROR: parameter type `A` may not live long enough
}

#[rustfmt::skip]
mod test_implied_from_fn_sig {
    type Opaque<T: 'static> = impl Sized;
    #[defines(Opaque<T>)]
    fn defining<T: 'static>() -> Opaque<T> {}
    fn assert_static<T: 'static>() {}
    fn test<T>(_: Opaque<T>) { assert_static::<T>(); }
}

fn main() {}
