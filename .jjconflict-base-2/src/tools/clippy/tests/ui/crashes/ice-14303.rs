//@check-pass
#![warn(clippy::macro_use_imports)]

#[repr(transparent)]
pub struct X(());

#[repr(u8)]
pub enum Action {
    Off = 0,
}

fn main() {}
