#![feature(const_impl_trait)]
// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait, impl_trait_in_bindings))]
//[full_tait]~^ WARN incomplete
//[full_tait]~| WARN incomplete

type Bar = impl Send;

// While i32 is structural-match, we do not want to leak this information.
// (See https://github.com/rust-lang/rust/issues/72156)
const fn leak_free() -> Bar {
    7i32
}
const LEAK_FREE: Bar = leak_free(); //[min_tait]~ ERROR not permitted here

fn leak_free_test() {
    match todo!() {
        LEAK_FREE => (),
        //[full_tait]~^ `impl Send` cannot be used in patterns
        _ => (),
    }
}

fn main() {}
