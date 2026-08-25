//@ incremental
//@ compile-flags: -Copt-level=0

#![crate_type = "lib"]

//@ aux-build:cgu_generic_function.rs
extern crate cgu_generic_function;

// This test checks that, in an unoptimized build, a generic function and its callees are only
// instantiated once in this crate.

//~ MONO_ITEM fn user @@ extern_generic[External]
pub fn user() {
    let _ = cgu_generic_function::foo("abc");
}

pub mod mod1 {
    use cgu_generic_function;

    //~ MONO_ITEM fn mod1::user @@ extern_generic-mod1[External]
    pub fn user() {
        let _ = cgu_generic_function::foo("abc");
    }

    pub mod mod1 {
        use cgu_generic_function;

        //~ MONO_ITEM fn mod1::mod1::user @@ extern_generic-mod1-mod1[External]
        pub fn user() {
            let _ = cgu_generic_function::foo("abc");
        }
    }
}

//~ MONO_ITEM fn cgu_generic_function::foo::<&str> @@ cgu_generic_function-in-extern_generic.volatile[External]
//~ MONO_ITEM fn cgu_generic_function::bar::<&str> @@ cgu_generic_function-in-extern_generic.volatile[External]
