//@ compile-flags: --remap-path-prefix={{src-base}}=remapped
//@ compile-flags: --remap-path-scope=debuginfo

#[macro_export]
macro_rules! my_file {
    () => { file!() }
}

pub fn file() -> &'static str {
    file!()
}
