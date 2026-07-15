//@ edition: 2018

#[macro_export]
macro_rules! import_all {
    () => {
        use macro_source::*;
    };
}

#[macro_export]
macro_rules! macro_use_source {
    () => {
        #[macro_use]
        extern crate macro_source;
    };
}

#[macro_export]
macro_rules! call_redirected_trait {
    () => {
        ().redirected_method()
    };
}

#[macro_export]
macro_rules! redirected_type {
    () => {
        RedirectedItem
    };
}

#[macro_export]
macro_rules! redirected_value {
    () => {
        RedirectedItem
    };
}
