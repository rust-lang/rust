//@ edition:2015

#![feature(view_types, view_type_macro)]

pub use std::view::view_type;

#[macro_export]
macro_rules! view_bar {
    () => {
        $crate::view_type!(Bar.{ async })
    }
}
