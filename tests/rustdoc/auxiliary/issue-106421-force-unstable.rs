//@ compile-flags: -Zforce-unstable-if-unmarked
#![crate_name="foo"]
pub struct FatalError;

impl FatalError {
    pub fn raise(self) -> ! {
        loop {}
    }
}
