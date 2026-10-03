//@ compile-flags: --remap-path-prefix={{src-base}}=remapped
//@ compile-flags: --remap-path-scope=documentation -Zunstable-options

#[macro_export]
macro_rules! my_file {
    () => {
        file!()
    };
}

pub fn file() -> &'static str {
    file!()
}
