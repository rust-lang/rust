pub unsafe fn unsf() {}

#[macro_export]
macro_rules! unsafe_macro { () => ($crate::unsf()) }
