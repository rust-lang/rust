#![feature(type_alias_impl_trait)]

mod test_lifetime_param {
    type Ty<'a> = impl Sized;
    fn defining(a: &str) -> Ty<'_> { a }
    fn assert_static<'a: 'static>() {}
    //~^ WARN: unnecessary lifetime parameter `'a`
    fn test<'a>() where Ty<'a>: 'static { assert_static::<'a>() }
}

mod test_higher_kinded_lifetime_param {
    type Ty<'a> = impl Sized;
    fn defining(a: &str) -> Ty<'_> { a }
    fn assert_static<'a: 'static>() {}
    //~^ WARN: unnecessary lifetime parameter `'a`
    fn test<'a>() where for<'b> Ty<'b>: 'a { assert_static::<'a>() }
}

mod test_higher_kinded_lifetime_param2 {
    fn assert_static<'a: 'static>() {}
    //~^ WARN: unnecessary lifetime parameter `'a`
    fn test<'a>() { assert_static::<'a>() }
    // no error because all the other errors happen first and then we abort before
    // emitting an error here.
}

mod test_type_param {
    type Ty<A> = impl Sized;
    fn defining<A>(s: A) -> Ty<A> { s }
    fn assert_static<A: 'static>() {}
    fn test<A>() where Ty<A>: 'static { assert_static::<A>() }
}

mod test_type_param_static {
    type Ty<A> = impl Sized + 'static;
    //~^ ERROR: the parameter type `A` may not live long enough
    fn defining<A: 'static>(s: A) -> Ty<A> { s }
    fn assert_static<A: 'static>() {}
    fn test<A>() where Ty<A>: 'static { assert_static::<A>() }
}

fn main() {}
