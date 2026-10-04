//@ edition:2018

#[macro_export]
macro_rules! gen_imports { () => {
    use import::Path;
    use std::collections::LinkedList;

    fn check_absolute() {
        ::absolute::Path;
        ::std::collections::LinkedList::<u8>::new();
    }
}}

#[macro_export]
macro_rules! gen_glob { () => {
    use *;
}}
