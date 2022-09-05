#![feature(type_alias_impl_trait)]

type Bar = impl Send;

// While i32 is structural-match, we do not want to leak this information.
// (See https://github.com/rust-lang/rust/issues/72156)
const fn leak_free() -> Bar {
    7i32
}
const LEAK_FREE: Bar = leak_free();

fn leak_free_test() {
    match LEAK_FREE {
        LEAK_FREE => (),
        //~^ ERROR constant pattern depends on a generic parameter
        //~| ERROR constant pattern depends on a generic parameter
        _ => (),
    }
}

fn main() {}
