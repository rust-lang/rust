//@ build-pass (FIXME(62277): could be check-pass?)
//@ compile-flags:--extern extern_prelude --extern Vec
//@ aux-build:extern-prelude.rs
//@ aux-build:extern-prelude-vec.rs

fn basic() {
    // It works
    let s = extern_prelude::S;
    s.external();
}

fn shadow_mod() {
    // Local module shadows `extern_prelude` from extern prelude
    mod extern_prelude {
        pub struct S;

        impl S {
            pub fn internal(&self) {}
        }
    }

    let s = extern_prelude::S;
    s.internal(); // OK
}

fn shadow_prelude() {
    // Extern prelude shadows standard library prelude
    let x: () = Vec::new(0f32, ()); // OK
}

fn main() {}
