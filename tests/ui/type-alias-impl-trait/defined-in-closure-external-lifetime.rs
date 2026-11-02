//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
#![feature(type_alias_impl_trait)]

mod case1 {
    type Opaque<'x> = impl Sized + 'x;
    #[define_opaque(Opaque)]
    fn foo<'s>() {
        let _ = || { let _: Opaque<'s> = (); };
        //[current]~^ ERROR expected generic lifetime parameter, found `'_`
        //[next]~^^ ERROR non-defining use of `case1::Opaque<'_>` in the defining scope
    }
}

mod case2 {
    type Opaque<'x> = impl Sized + 'x;
    #[define_opaque(Opaque)]
    fn foo<'s>() {
        let _ = || -> Opaque<'s> {};
        //[current]~^ ERROR expected generic lifetime parameter, found `'_`
        //[next]~^^ ERROR non-defining use of `case2::Opaque<'_>` in the defining scope
    }
}

fn main() {}
