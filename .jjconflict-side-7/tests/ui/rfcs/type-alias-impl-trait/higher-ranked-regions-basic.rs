// Basic tests for opaque type inference under for<_> binders.

#![feature(type_alias_impl_trait)]

trait Trait<'a> {
    type Ty;
}
impl<'a, T> Trait<'a> for T {
    type Ty = &'a ();
}

mod basic_pass {
    use super::*;
    type Opq<'a> = impl Sized + 'a;
    #[define_opaque(Opq)]
    fn test() -> impl for<'a> Trait<'a, Ty = Opq<'a>> {}
    //~^ ERROR: expected generic lifetime parameter, found `'a`
}

mod capture_rpit {
    use super::*;
    fn test() -> impl for<'a> Trait<'a, Ty = impl Sized> {}
    //~^ ERROR hidden type for `impl Sized` captures lifetime that does not appear in bounds
}

mod capture_tait {
    use super::*;
    type Opq0 = impl Sized;
    type Opq1<'a> = impl for<'b> Trait<'b, Ty = Opq0>;
    type Opq2 = impl for<'a> Trait<'a, Ty = Opq1<'a>>;
    #[define_opaque(Opq2)]
    fn test() -> Opq2 {}
    //~^ ERROR hidden type for `capture_tait::Opq0` captures lifetime that does not appear in bounds
    //~| ERROR expected generic lifetime parameter, found `'a`
}

mod capture_tait_complex_pass {
    use super::*;
    type Opq0<'a> = impl Sized;
    type Opq1<'a> = impl for<'b> Trait<'b, Ty = Opq0<'b>>; // <- Note 'b
    type Opq2 = impl for<'a> Trait<'a, Ty = Opq1<'a>>;
    #[define_opaque(Opq2)]
    fn test() -> Opq2 {}
    //~^ ERROR: expected generic lifetime parameter, found `'a`
    //~| ERROR: expected generic lifetime parameter, found `'b`
}

// Same as the above, but make sure that different placeholder regions are not equal.
mod capture_tait_complex_fail {
    use super::*;
    type Opq0<'a> = impl Sized;
    type Opq1<'a> = impl for<'b> Trait<'b, Ty = Opq0<'a>>; // <- Note 'a
    type Opq2 = impl for<'a> Trait<'a, Ty = Opq1<'a>>;
    #[define_opaque(Opq2)]
    fn test() -> Opq2 {}
    //~^ ERROR expected generic lifetime parameter, found `'a`
    //~| ERROR expected generic lifetime parameter, found `'a`
}

// non-defining use because 'static is used.
mod constrain_fail0 {
    use super::*;
    type Opq0<'a, 'b> = impl Sized;
    #[define_opaque(Opq0)]
    fn test() -> impl for<'a> Trait<'a, Ty = Opq0<'a, 'static>> {}
    //~^ ERROR: expected generic lifetime parameter, found `'a`
}

// non-defining use because generic lifetime is used multiple times.
mod constrain_fail {
    use super::*;
    type Opq0<'a, 'b> = impl Sized;
    #[define_opaque(Opq0)]
    fn test() -> impl for<'a> Trait<'a, Ty = Opq0<'a, 'a>> {}
    //~^ ERROR: expected generic lifetime parameter, found `'a`
}

mod constrain_pass {
    use super::*;
    type Opq0<'a, 'b> = impl Sized;
    type Opq1<'a> = impl for<'b> Trait<'b, Ty = Opq0<'a, 'b>>;
    type Opq2 = impl for<'a> Trait<'a, Ty = Opq1<'a>>;
    #[define_opaque(Opq2)]
    fn test() -> Opq2 {}
    //~^ ERROR: expected generic lifetime parameter, found `'a`
    //~| ERROR: expected generic lifetime parameter, found `'a`
}

fn main() {}
