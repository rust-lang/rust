//@ check-pass
//@ reference: cfg.target_vendor.def
//@ reference: cfg.target_vendor.values
#[cfg(target_vendor = "unknown")]
pub fn main() {
}

#[cfg(not(target_vendor = "unknown"))]
pub fn main() {
}
