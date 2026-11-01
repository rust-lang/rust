#![crate_type = "lib"]

#[cfg(overflow_checks)] //~ ERROR `cfg(overflow_checks)` is experimental
pub fn cast(v: i64)->u32{
    todo!()
}
