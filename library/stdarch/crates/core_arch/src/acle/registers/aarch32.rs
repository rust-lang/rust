/// Application Program Status Register
pub struct APSR;

#[cfg(any(not(target_feature = thumb-mode), target_feature = "v6t2"))]
rsr!(APSR);
