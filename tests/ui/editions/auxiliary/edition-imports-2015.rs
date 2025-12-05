//@ edition:2015

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

#[macro_export]
macro_rules! gen_gated { () => {
    fn check_gated() {
        enum E { A }
        use E::*;
    }
}}

#[macro_export]
macro_rules! gen_ambiguous { () => {
    use Ambiguous;
    type A = ::edition_imports_2015::Path;
}}
