// If the hidden type is a closure, we require the "outlives" bounds that appear on the
// defining site to also appear on the opaque type.
//
// It's not clear if this is the desired behavior but at least
// it's consistent and has no back-compat risk.

// check-fail

#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

// requires `'a: 'b` bound
mod test1 {
    type Opaque<'a, 'b> = impl Sized + 'a + 'b;
    //~^ ERROR lifetime bound not satisfied

    fn define<'a, 'b>() -> Opaque<'a, 'b>
    where
        'a: 'b,
    {
        || {}
    }
}

// Same as the above but through indirection `'x`
mod test2 {
    type Opaque<'a, 'b> = impl Sized + 'a + 'b;
    //~^ ERROR cannot infer an appropriate lifetime

    fn define<'a, 'b, 'x>() -> Opaque<'a, 'b>
    where
        'a: 'x,
        'x: 'b,
    {
        || {}
    }
}

// fixed version of the above
mod test2_fixed {
    type Opaque<'a: 'b, 'b> = impl Sized + 'a + 'b;

    fn define<'a, 'b, 'x>() -> Opaque<'a, 'b>
    where
        'a: 'x,
        'x: 'b,
    {
        || {}
    }
}

// requires `T: 'static`
mod test3 {
    type Opaque<T> = impl Sized;
    //~^ ERROR the parameter type `T` may not live long enough

    fn define<T>() -> Opaque<T>
    where
        T: 'static,
    {
        || {}
    }
}

fn main() {}
