#![feature(rustc_attrs)]

#[rustc_on_unimplemented(
    message = "normal: {This}, path: {This:path},  resolved: {This:resolved}"
)]
pub trait Trait<'lifetime, const CONST_GENERIC: usize, A, B> where A: Send {}

fn take_trait<'a, T: Trait<'a, 6, u8, U>, U>(_: T) {}

fn main() {
    take_trait(());
    //~^ERROR normal: Trait, path: Trait<'lifetime, CONST_GENERIC, A, B>,  resolved: Trait<'_, 6, u8, _>
}
