// ignore-tidy-linelength
// We specify -Z incremental here because we want to test the partitioning for
// incremental compilation
// compile-flags:-Zprint-mono-items=eager -Zincremental=tmp/partitioning-tests/local-generic

#![allow(dead_code)]
#![crate_type="lib"]

//~ MONO_ITEM fn local_generic::generic[0]<u32> @@ local_generic.volatile[External]
//~ MONO_ITEM fn local_generic::generic[0]<u64> @@ local_generic.volatile[External]
//~ MONO_ITEM fn local_generic::generic[0]<char> @@ local_generic.volatile[External]
//~ MONO_ITEM fn local_generic::generic[0]<&str> @@ local_generic.volatile[External]
pub fn generic<T>(x: T) -> T { x }

//~ MONO_ITEM fn local_generic::user[0] @@ local_generic[Internal]
fn user() {
    let _ = generic(0u32);
}

mod mod1 {
    pub use super::generic;

    //~ MONO_ITEM fn local_generic::mod1[0]::user[0] @@ local_generic-mod1[Internal]
    fn user() {
        let _ = generic(0u64);
    }

    mod mod1 {
        use super::generic;

        //~ MONO_ITEM fn local_generic::mod1[0]::mod1[0]::user[0] @@ local_generic-mod1-mod1[Internal]
        fn user() {
            let _ = generic('c');
        }
    }
}

mod mod2 {
    use super::generic;

    //~ MONO_ITEM fn local_generic::mod2[0]::user[0] @@ local_generic-mod2[Internal]
    fn user() {
        let _ = generic("abc");
    }
}
