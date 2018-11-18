// edition:2015

#[macro_export]
macro_rules! gen_imports { () => {
    use import::Path;
    // use std::collections::LinkedList; // FIXME

    fn check_absolute() {
        ::absolute::Path;
        // ::std::collections::LinkedList::<u8>::new(); // FIXME
    }
}}

#[macro_export]
macro_rules! gen_glob { () => {
    use *;
}}
