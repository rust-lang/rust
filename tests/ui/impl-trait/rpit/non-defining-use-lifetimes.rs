// issue: #111935

#![allow(unconditional_recursion)]

// Lt indirection is necessary to make the lifetime of the function late-bound,
// in order to bypass some other bugs.
type Lt<'lt> = Option<*mut &'lt ()>;

mod statik {
    use super::*;
    // invalid defining use: Opaque<'static> := ()
    fn foo<'a>(_: Lt<'a>) -> impl Sized + 'a {
        let _: () = foo(Lt::<'static>::None);
        //~^ ERROR expected generic lifetime parameter, found `'static`
    }
}

mod infer {
    use super::*;
    // invalid defining use: Opaque<'_> := ()
    fn foo<'a>(_: Lt<'a>) -> impl Sized + 'a {
        let _: () = foo(Lt::<'_>::None);
        //~^ ERROR expected generic lifetime parameter, found `'_`
    }
}

mod equal {
    use super::*;
    // invalid defining use: Opaque<'a, 'a> := ()
    // because of the use of equal lifetimes in args
    fn foo<'a, 'b>(_: Lt<'a>, _: Lt<'b>) -> impl Sized + 'a + 'b {
        let _: () = foo(Lt::<'a>::None, Lt::<'a>::None);
        //~^ ERROR non-defining opaque type use in defining scope
    }
}

fn main() {}
