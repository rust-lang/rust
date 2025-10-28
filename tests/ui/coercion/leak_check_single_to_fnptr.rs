//@ revisions: livecode deadcode

// When coercing to a fnptr check that we leak check when relating
// the signatures. Previously we only leak checked for fnptr/fndef
// to fnptr coercions, but not closure to fnptr coercions. This
// resulted in closure to fnptr coercions not erroring in dead code
// when equivalent code with fndefs/fnptrs would error.
//
// Outside of dead code all cases wind up erroring in borrowck so this
// is not a soundness concern.

fn mk<T>() -> T {
    loop {}
}

fn static_fndef(_: &'static ()) {}

fn one_way_coerce() {
    let a_fndef = static_fndef;
    let a_fnptr = mk::<fn(&'static ())>();
    let a_closure = |_: &'static ()| {};

    #[cfg(deadcode)]
    loop {}
    let _target: for<'a> fn(&'a ()) = a_fndef;
    //[livecode]~^ ERROR: mismatched types
    //[deadcode]~^^ ERROR: mismatched types
    let _target: for<'a> fn(&'a ()) = a_fnptr;
    //[livecode]~^ ERROR: mismatched types
    //[deadcode]~^^ ERROR: mismatched types
    let _target: for<'a> fn(&'a ()) = a_closure;
    //[livecode]~^ ERROR: mismatched types
    //[deadcode]~^^ ERROR: mismatched types
}

fn main() {}
