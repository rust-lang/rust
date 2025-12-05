pub type S = u8;

macro_rules! generate_exported { () => {
    #[macro_export]
    macro_rules! exported {
        () => ($crate::S)
    }
}}

generate_exported!();
