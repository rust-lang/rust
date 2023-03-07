#[macro_export]
macro_rules! attrs_on_struct {
    ( $( #[$attr:meta] )* ) => {
        $( #[$attr] )*
        pub struct ExpandedStruct;
    }
}
