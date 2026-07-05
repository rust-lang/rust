//! Test that `#[splat]` on `&dyn AsRef<T>` where `T: Tuple` is an error.

#![allow(incomplete_features)]
#![feature(splat)]
#![feature(tuple_trait)]

// Strip binders and their lifetime numbers from error messages
//@ normalize-stderr: "&.*value: (.*), bound_vars: .*" -> "$1"

// FIXME(splat): Some errors are reported on the callee, but they would be more ergonomic on the
// caller as well
fn dyn_asref_splat<T>(#[splat] _t: &dyn AsRef<T>)
//~^ ERROR cannot use splat attribute; the splatted argument type must be a tuple or unit, not a
//~| ERROR cannot use splat attribute; the splatted argument type must be a tuple or unit, not a
//~| ERROR cannot use splat attribute; the splatted argument type must be a tuple or unit, not a
//~| ERROR cannot use splat attribute; the splatted argument type must be a tuple or unit, not a
where
    T: std::marker::Tuple,
{
}

fn main() {
    // These error patterns are reported on the function definition, but we can't check for two
    // strings in the same error message
    let s: String = "hello".to_owned();
    dyn_asref_splat::<String>(&s);
    //~^ ERROR `String` is not a tuple
    //@regex-error-pattern: type must be a tuple or unit, not a .* Trait\(.*AsRef<.*String>\)

    dyn_asref_splat(&s);
    //@regex-error-pattern: type must be a tuple or unit, not a .* Trait\(.*AsRef<_>\)

    let t = (1u8, 2f32);
    dyn_asref_splat::<(u8, f32)>(&t);
    //@regex-error-pattern: type must be a tuple or unit, not a .* Trait\(.*AsRef<\(u8, f32\)>\)

    dyn_asref_splat(&t);
    //@regex-error-pattern: type must be a tuple or unit, not a .* Trait\(.*AsRef<_>\)
}
