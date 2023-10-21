// issue: #111935
// FIXME(aliemjay): outdated due to "once modulo regions" restriction.
// FIXME(aliemjay): mod `infer` should fail.

#![allow(unconditional_recursion)]

// Lt indirection is necessary to make the lifetime of the function late-bound,
// in order to bypass some other bugs.
type Lt<'lt> = Option<*mut &'lt ()>;

mod statik {
    use super::*;
    // invalid defining use: Opaque<'static> := ()
    fn foo<'a>(_: Lt<'a>) -> impl Sized + 'a {
        let _: () = foo(Lt::<'static>::None);
        //~^ ERROR opaque type used twice with different lifetimes
    }
}

mod infer {
    use super::*;
    // invalid defining use: Opaque<'_> := ()
    fn foo<'a>(_: Lt<'a>) -> impl Sized + 'a {
        let _: () = foo(Lt::<'_>::None);
    }
}

mod equal {
    use super::*;
    // invalid defining use: Opaque<'a, 'a> := ()
    // because of the use of equal lifetimes in args
    fn foo<'a, 'b>(_: Lt<'a>, _: Lt<'b>) -> impl Sized + 'a + 'b {
        let _: () = foo(Lt::<'a>::None, Lt::<'a>::None);
        //~^ ERROR opaque type used twice with different lifetimes
    }
}

fn main() {}
