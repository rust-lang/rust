#![feature(const_trait_impl)]
#![feature(staged_api)]
#![stable(feature = "rust1", since = "1.0.0")]

#[stable(feature = "rust1", since = "1.0.0")]
pub struct Probe;

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature = "const_trait_gate", issue = "none")]
pub const trait TraitWithConstGate {
    #[stable(feature = "rust1", since = "1.0.0")]
    const ASSOC: usize;
}

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature = "const_trait_gate", issue = "none")]
const impl TraitWithConstGate for Probe {
    const ASSOC: usize = 1;
}
