#![feature(type_alias_impl_trait)]

mod case1 {
    type Opaque<'x> = impl Sized + 'x;
    #[define_opaque(Opaque)]
    fn foo<'s>() -> Opaque<'s> {
        let _ = || { let _: Opaque<'s> = (); };
        //~^ ERROR expected generic lifetime parameter, found `'_`
    }
}

mod case2 {
    type Opaque<'x> = impl Sized + 'x;
    #[define_opaque(Opaque)]
    fn foo<'s>() -> Opaque<'s> {
        let _ = || -> Opaque<'s> {};
        //~^ ERROR expected generic lifetime parameter, found `'_`
    }
}

fn main() {}
