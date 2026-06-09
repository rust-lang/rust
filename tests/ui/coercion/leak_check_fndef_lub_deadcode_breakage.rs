//@ normalize-stderr: "32 bits" -> "64 bits"

fn foo<T>() {}

fn fndef_lub_leak_check() {
    macro_rules! lub {
        ($lhs:expr, $rhs:expr) => {
            if true { $lhs } else { $rhs }
        };
    }

    // Unused parameters on FnDefs are considered invariant
    let lhs = foo::<for<'a> fn(&'static (), &'a ())>;
    let rhs = foo::<for<'a> fn(&'a (), &'static ())>;

    loop {}

    // If we leak check then we know we should coerce these
    // to `fn()`, if we don't leak check we may try to keep
    // them as `FnDef`s which would cause this code to compile
    // as borrowck won't emit errors for deadcode.
    let lubbed = lub!(lhs, rhs);

    // assert that `lubbed` is a ZST/`FnDef`
    unsafe { std::mem::transmute::<_, ()>(lubbed) }
    //~^ ERROR: cannot transmute between types of different sizes
}

fn main() {}
