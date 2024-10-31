//@ edition:2024
//@ compile-flags:-Z unstable-options
#![crate_name="extern_macros"]
#[macro_export]
macro_rules! attrs_on_struct {
    ( $( #[$attr:meta] )* ) => {
        $( #[$attr] )*
        pub struct ExpandedStruct;
    }
}
