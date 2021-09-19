//
// We specify incremental here because we want to test the partitioning for
// incremental compilation
// incremental
// compile-flags:-Zprint-mono-items=eager -Zshare-generics=y

#![allow(dead_code)]
#![crate_type="lib"]

// aux-build:cgu_generic_function.rs
extern crate cgu_generic_function;

//~ MONO_ITEM fn user @@ extern_generic[Internal]
fn user() {
    let _ = cgu_generic_function::foo("abc");
}

mod mod1 {
    use cgu_generic_function;

    //~ MONO_ITEM fn mod1::user @@ extern_generic-mod1[Internal]
    fn user() {
        let _ = cgu_generic_function::foo("abc");
    }

    mod mod1 {
        use cgu_generic_function;

        //~ MONO_ITEM fn mod1::mod1::user @@ extern_generic-mod1-mod1[Internal]
        fn user() {
            let _ = cgu_generic_function::foo("abc");
        }
    }
}

mod mod2 {
    use cgu_generic_function;

    //~ MONO_ITEM fn mod2::user @@ extern_generic-mod2[Internal]
    fn user() {
        let _ = cgu_generic_function::foo("abc");
    }
}

mod mod3 {
    //~ MONO_ITEM fn mod3::non_user @@ extern_generic-mod3[Internal]
    fn non_user() {}
}

// Make sure the two generic functions from the extern crate get instantiated
// once for the current crate
//~ MONO_ITEM fn cgu_generic_function::foo::<&str> @@ cgu_generic_function-in-extern_generic.volatile[External]
//~ MONO_ITEM fn cgu_generic_function::bar::<&str> @@ cgu_generic_function-in-extern_generic.volatile[External]
