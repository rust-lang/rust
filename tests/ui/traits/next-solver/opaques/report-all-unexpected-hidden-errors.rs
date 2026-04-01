//@ compile-flags: -Znext-solver

// Just for diagnostics completeness.
// This is probably unimportant as we only report one error for such case in HIR typeck.

#![feature(type_alias_impl_trait)]

struct Invar<'a>(*mut &'a ());

fn mk_invar<'a>(a: &'a i32) -> Invar<'a> {
    todo!()
}

type MultiUse = impl Sized;

#[define_opaque(MultiUse)]
fn capture_different_universals_not_on_bounds<'a, 'b, 'c>(a: &'a i32, b: &'b i32, c: &'c i32)  {
    let _ = || -> MultiUse {
        //~^ ERROR: hidden type for `MultiUse` captures lifetime that does not appear in bounds [E0700]
        mk_invar(a)
    };
    let _ = || -> MultiUse {
        //~^ ERROR: hidden type for `MultiUse` captures lifetime that does not appear in bounds [E0700]
        mk_invar(b)
    };
    let _ = || {
        let _ = || -> MultiUse {
            //~^ ERROR: hidden type for `MultiUse` captures lifetime that does not appear in bounds [E0700]
            mk_invar(c)
        };
    };
}

fn main() {}
