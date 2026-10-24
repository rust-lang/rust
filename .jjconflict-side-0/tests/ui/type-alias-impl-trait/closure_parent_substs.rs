// When WF checking the hidden type in the ParamEnv of the opaque type,
// one complication arises when the hidden type is a closure/coroutine:
// the "parent_substs" of the type may reference lifetime parameters
// not present in the opaque type.
// These region parameters are not really useful in this check.
// So here we ignore them and replace them with fresh region variables.

//@ check-pass

#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

// Basic test
mod test1 {
    // Hidden type = Closure['?0]
    type Opaque = impl Sized;

    #[define_opaque(Opaque)]
    fn define<'a: 'a>() -> Opaque {
        || {}
    }
}

// the region vars cannot both be equal to `'static` or `'empty`
mod test2 {
    trait Trait {}

    // Hidden type = Closure['a, '?0, '?1]
    // Constraints = [('?0: 'a), ('a: '?1)]
    type Opaque<'a>
    where
        &'a (): Trait,
    = impl Sized + 'a;

    #[define_opaque(Opaque)]
    fn define<'a, 'x, 'y>() -> Opaque<'a>
    where
        &'a (): Trait,
        'x: 'a,
        'a: 'y,
    {
        || {}
    }
}

// the region var cannot be equal to `'a` or `'b`
mod test3 {
    trait Trait {}

    // Hidden type = Closure['a, 'b, '?0]
    // Constraints = [('?0: 'a), ('?0: 'b)]
    type Opaque<'a, 'b>
    where
        (&'a (), &'b ()): Trait,
    = impl Sized + 'a + 'b;

    #[define_opaque(Opaque)]
    fn define<'a, 'b, 'x>() -> Opaque<'a, 'b>
    where
        (&'a (), &'b ()): Trait,
        'x: 'a,
        'x: 'b,
    {
        || {}
    }
}

fn main() {}
