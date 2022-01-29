pub trait Tr {
    // Note: The function needs to be declared over multiple lines to reproduce
    // the crash. DO NOT reformat.
    fn f()
        where Self: Sized;
}
