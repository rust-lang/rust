//@ check-pass

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

    // If we leak check then we know we should coerce these
    // to `fn()`, if we don't leak check we may try to keep
    // them as `FnDef`s which would result in a borrowck
    // error.
    let lubbed = lub!(lhs, rhs);

    // assert that we coerced lhs/rhs to a fn ptr
    is_fnptr(lubbed);
}

trait FnPtr {}
impl FnPtr for fn() {}
fn is_fnptr<T: FnPtr>(_: T) {}

fn main() {}
