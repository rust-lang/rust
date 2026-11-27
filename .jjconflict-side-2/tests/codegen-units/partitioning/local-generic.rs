//@ incremental
//@ compile-flags: -Copt-level=0

#![crate_type = "lib"]

// This test checks that all the instantiations of a local generic fn are placed in the same CGU,
// regardless of where it is called.

//~ MONO_ITEM fn generic::<u32> @@ local_generic.volatile[External]
//~ MONO_ITEM fn generic::<u64> @@ local_generic.volatile[External]
//~ MONO_ITEM fn generic::<char> @@ local_generic.volatile[External]
//~ MONO_ITEM fn generic::<&str> @@ local_generic.volatile[External]
pub fn generic<T>(x: T) -> T {
    x
}

//~ MONO_ITEM fn user @@ local_generic[External]
pub fn user() {
    let _ = generic(0u32);
}

pub mod mod1 {
    pub use super::generic;

    //~ MONO_ITEM fn mod1::user @@ local_generic-mod1[External]
    pub fn user() {
        let _ = generic(0u64);
    }

    pub mod mod1 {
        use super::generic;

        //~ MONO_ITEM fn mod1::mod1::user @@ local_generic-mod1-mod1[External]
        pub fn user() {
            let _ = generic('c');
        }
    }
}

pub mod mod2 {
    use super::generic;

    //~ MONO_ITEM fn mod2::user @@ local_generic-mod2[External]
    pub fn user() {
        let _ = generic("abc");
    }
}
