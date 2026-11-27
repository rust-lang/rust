// Test which ensures that attribute macros are correctly handled by the search.
// For example: `macro1` should appear in both `attr` and `macro` filters whereas
// `macro2` and `macro3` should only appear in `attr` or `macro` filters respectively.

#![feature(macro_attr)]
#![feature(macro_derive)]

#[macro_export]
macro_rules! macro1 {
    attr() () => {};
    derive() () => {};
    () => {};
}

#[macro_export]
macro_rules! macro2 {
    attr() () => {};
}

#[macro_export]
macro_rules! macro3 {
    () => {};
}

#[macro_export]
macro_rules! macro4 {
    derive() () => {};
}

#[macro_export]
macro_rules! macro5 {
    attr() () => {};
    derive() () => {};
}
