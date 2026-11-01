//@ edition:2018

#![allow(keyword_idents)]

#[macro_export]
macro_rules! produces_async {
    () => (pub fn async() {})
}

#[macro_export]
macro_rules! produces_async_raw {
    () => (pub fn r#async() {})
}

#[macro_export]
macro_rules! consumes_async {
    (async) => (1)
}

#[macro_export]
macro_rules! consumes_async_raw {
    (r#async) => (1)
}

#[macro_export]
macro_rules! passes_ident {
    ($i: ident) => ($i)
}

#[macro_export]
macro_rules! passes_tt {
    ($i: tt) => ($i)
}
