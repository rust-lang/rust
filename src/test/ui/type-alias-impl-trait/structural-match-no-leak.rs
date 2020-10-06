#![feature(const_impl_trait, type_alias_impl_trait)]

type Bar = impl Send;

// While i32 is structural-match, we do not want to leak this information.
// (See https://github.com/rust-lang/rust/issues/72156)
const fn leak_free() -> Bar {
    7i32
}
const LEAK_FREE: Bar = leak_free();

fn leak_free_test() {
    match todo!() {
        LEAK_FREE => (),
        //~^ `impl Send` cannot be used in patterns
        _ => (),
    }
}

fn main() {}
