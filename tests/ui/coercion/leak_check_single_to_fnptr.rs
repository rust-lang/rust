//@ revisions: livecode deadcode

// When coercing to a fnptr check that we leak check when relating
// the signatures. Previously we would leak checked for fnptr/fndef
// to fnptr coercions, but not closure to fnptr coercions. This
// resulted in closure to fnptr coercions not erroring in dead code
// when equivalent code with fndefs/fnptrs would error.
//
// We now never leak check during normal subtyping operations only
// during lub operations. Outside of dead code all cases wind up
// erroring in borrowck so this is not a soundness concern.

fn mk<T>() -> T {
    loop {}
}

fn static_fndef(_: &'static ()) {}

fn one_way_coerce() {
    let a_fndef = static_fndef;
    let a_fnptr = mk::<fn(&'static ())>();

    #[cfg(deadcode)]
    loop {}
    let _target: for<'a> fn(&'a ()) = a_fndef;
    //[livecode]~^ ERROR: mismatched types
    //[deadcode]~^^ ERROR: mismatched types
    let _target: for<'a> fn(&'a ()) = a_fnptr;
    //[livecode]~^ ERROR: mismatched types
    //[deadcode]~^^ ERROR: mismatched types
}

fn one_way_coerce_2() {
    let a_closure = |_: &'static ()| {};

    #[cfg(deadcode)]
    loop {}

    let _target: for<'a> fn(&'a ()) = a_closure;
    //[livecode]~^ ERROR: mismatched types

}

fn main() {}
