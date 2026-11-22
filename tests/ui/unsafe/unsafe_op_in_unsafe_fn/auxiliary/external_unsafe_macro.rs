pub unsafe fn unsf() {}

#[macro_export]
macro_rules! unsafe_macro { () => ($crate::unsf()) }

#[macro_export]
macro_rules! deref_ptr { ($p:expr) => { *$p } }
