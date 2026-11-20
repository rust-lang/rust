//@ compile-flags: -Znext-solver
//@ check-pass

// Fixes a regression in icu_provider_adapters where we weren't normalizing the
// return type of a function type before performing a `Ty::builtin_deref` call,
// leading to an ICE.

struct Struct {
    field: i32,
}

fn hello(f: impl Fn() -> &'static Box<[i32]>, f2: impl Fn() -> &'static Struct) {
    let cl = || {
        let x = &f()[0];
        let y = &f2().field;
    };
}

fn main() {}
