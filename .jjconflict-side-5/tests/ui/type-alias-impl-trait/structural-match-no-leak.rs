#![feature(type_alias_impl_trait)]

pub type Bar = impl Send;

// While i32 is structural-match, we do not want to leak this information.
// (See https://github.com/rust-lang/rust/issues/72156)
#[define_opaque(Bar)]
pub const fn leak_free() -> Bar {
    7i32
}

const LEAK_FREE: Bar = leak_free();

fn leak_free_test() {
    match LEAK_FREE {
        LEAK_FREE => (),
        //~^ ERROR `Bar` cannot be used in patterns
        _ => (),
    }
}

fn main() {}
