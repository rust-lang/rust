// Test that removing an upstream crate does not cause any trouble.

// revisions:rpass1 rpass2
// aux-build:extern_crate.rs

#[cfg(rpass1)]
extern crate extern_crate;

pub fn main() {
    #[cfg(rpass1)]
    {
        extern_crate::foo(1);
    }

    #[cfg(rpass2)]
    {
        foo(1);
    }
}

#[cfg(rpass2)]
pub fn foo(_: u8) {

}
