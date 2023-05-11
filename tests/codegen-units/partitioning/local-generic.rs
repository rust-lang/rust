// We specify incremental here because we want to test the partitioning for
// incremental compilation
// incremental
// compile-flags:-Zprint-mono-items=eager

#![allow(dead_code)]
#![crate_type="lib"]

//~ MONO_ITEM fn generic::<u32> @@ local_generic.volatile[External]
//~ MONO_ITEM fn generic::<u64> @@ local_generic.volatile[External]
//~ MONO_ITEM fn generic::<char> @@ local_generic.volatile[External]
//~ MONO_ITEM fn generic::<&str> @@ local_generic.volatile[External]
pub fn generic<T>(x: T) -> T { x }

//~ MONO_ITEM fn user @@ local_generic[Internal]
fn user() {
    let _ = generic(0u32);
}

mod mod1 {
    pub use super::generic;

    //~ MONO_ITEM fn mod1::user @@ local_generic-mod1[Internal]
    fn user() {
        let _ = generic(0u64);
    }

    mod mod1 {
        use super::generic;

        //~ MONO_ITEM fn mod1::mod1::user @@ local_generic-mod1-mod1[Internal]
        fn user() {
            let _ = generic('c');
        }
    }
}

mod mod2 {
    use super::generic;

    //~ MONO_ITEM fn mod2::user @@ local_generic-mod2[Internal]
    fn user() {
        let _ = generic("abc");
    }
}
