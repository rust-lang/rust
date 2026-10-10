//! Auxiliary file for <https://github.com/rust-lang/rust/issues/41652>.

pub trait Tr {
    // Note: The function needs to be declared over multiple lines to reproduce
    // the crash. DO NOT reformat.
    fn f()
        where Self: Sized;
}
