/// Application Program Status Register
pub struct APSR;

// Note (@Lokathor): Because this breaks the use of Rust on the Game Boy
// Advance, this change must be reverted until Rust learns to handle cpu state
// properly. See also: https://github.com/rust-lang/stdarch/issues/702

//#[cfg(any(not(target_feature = "thumb-state"), target_feature = "v6t2"))]
//rsr!(APSR);
