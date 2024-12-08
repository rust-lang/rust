#![feature(type_alias_impl_trait)]

mod bar {
    pub type Bar = impl Send;

    // While i32 is structural-match, we do not want to leak this information.
    // (See https://github.com/rust-lang/rust/issues/72156)
    pub const fn leak_free() -> Bar {
        7i32
    }
}
const LEAK_FREE: bar::Bar = bar::leak_free();

fn leak_free_test() {
    match LEAK_FREE {
        LEAK_FREE => (),
        //~^ `Bar` cannot be used in patterns
        _ => (),
    }
}

fn main() {}
