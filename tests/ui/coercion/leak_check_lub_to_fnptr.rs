//@ revisions: livecode deadcode

#![allow(unreachable_code)]

fn mk<T>() -> T {
    loop {}
}

macro_rules! lub {
    ($lhs:expr, $rhs:expr) => {
        if true { $lhs } else { $rhs }
    };
}

fn lub_to_fnptr_leak_checking() {
    #[cfg(deadcode)]
    loop {}

    let lhs_fnptr = mk::<fn(&(), &'static ())>();
    let rhs_fnptr = mk::<fn(&'static (), &())>();
    lub!(lhs_fnptr, rhs_fnptr);
    //[livecode]~^ ERROR: `if` and `else` have incompatible types
    //[deadcode]~^^ ERROR: `if` and `else` have incompatible types

    fn lhs_fndef(_: &(), _: &'static ()) {};
    fn rhs_fndef(_: &'static (), _: &()) {};
    lub!(lhs_fndef, rhs_fndef);
    //[livecode]~^ ERROR: `if` and `else` have incompatible types
    //[deadcode]~^^ ERROR: `if` and `else` have incompatible types

    let lhs_closure = |_: &(), _: &'static ()| {};
    let rhs_closure = |_: &'static (), _: &()| {};
    lub!(lhs_closure, rhs_closure);
    //[livecode]~^ ERROR: `if` and `else` have incompatible types
    //[deadcode]~^^ ERROR: `if` and `else` have incompatible types
}

fn order_dependence_closures() {
    let lhs_closure = |_: &(), _: &'static ()| {};
    let rhs_closure = |_: &'static (), _: &'static ()| {};

    lub!(lhs_closure, rhs_closure);
    //~^ ERROR: `if` and `else` have incompatible types
    lub!(rhs_closure, lhs_closure);
    //~^ ERROR: `if` and `else` have incompatible types

}


fn order_dependence_fndefs() {
    fn lhs_fndef(_: &(), _: &'static ()) {}
    fn rhs_fndef(_: &'static (), _: &'static ()) {}

    lub!(lhs_fndef, rhs_fndef);
    //~^ ERROR: `if` and `else` have incompatible types
    lub!(rhs_fndef, lhs_fndef);
    //~^ ERROR: `if` and `else` have incompatible types
}

fn main() {}
