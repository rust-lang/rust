//@ compile-flags: -Clink-dead-code -Csymbol-mangling-version=v0
//@ build-pass

// Ensure that when eagerly collecting `test::{closure#0}`, we don't try
// collecting an unnormalized version of the closure (specifically its
// upvars), since the closure captures the RPIT `opaque::{opaque#0}`.

fn opaque() -> impl Sized {}

fn test() -> impl FnOnce() {
    let opaque = opaque();
    move || {
        let opaque = opaque;
    }
}

fn main() {
    test()();
}
