// ignore-tidy-linelength
// We specify -Z incremental here because we want to test the partitioning for
// incremental compilation
// compile-flags:-Zprint-mono-items=eager -Zincremental=tmp/partitioning-tests/extern-generic -Zshare-generics=y

#![allow(dead_code)]
#![crate_type="lib"]

// aux-build:cgu_generic_function.rs
extern crate cgu_generic_function;

//~ MONO_ITEM fn extern_generic::user[0] @@ extern_generic[Internal]
fn user() {
    let _ = cgu_generic_function::foo("abc");
}

mod mod1 {
    use cgu_generic_function;

    //~ MONO_ITEM fn extern_generic::mod1[0]::user[0] @@ extern_generic-mod1[Internal]
    fn user() {
        let _ = cgu_generic_function::foo("abc");
    }

    mod mod1 {
        use cgu_generic_function;

        //~ MONO_ITEM fn extern_generic::mod1[0]::mod1[0]::user[0] @@ extern_generic-mod1-mod1[Internal]
        fn user() {
            let _ = cgu_generic_function::foo("abc");
        }
    }
}

mod mod2 {
    use cgu_generic_function;

    //~ MONO_ITEM fn extern_generic::mod2[0]::user[0] @@ extern_generic-mod2[Internal]
    fn user() {
        let _ = cgu_generic_function::foo("abc");
    }
}

mod mod3 {
    //~ MONO_ITEM fn extern_generic::mod3[0]::non_user[0] @@ extern_generic-mod3[Internal]
    fn non_user() {}
}

// Make sure the two generic functions from the extern crate get instantiated
// once for the current crate
//~ MONO_ITEM fn cgu_generic_function::foo[0]<&str> @@ cgu_generic_function-in-extern_generic.volatile[External]
//~ MONO_ITEM fn cgu_generic_function::bar[0]<&str> @@ cgu_generic_function-in-extern_generic.volatile[External]
