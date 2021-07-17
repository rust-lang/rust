// edition:2015

#![feature(rustc_attrs)]

#[rustc_per_edition]
pub type I32OrStr = (
    i32, // 2015
    &'static str, // 2018+
);

pub type I32 = I32OrStr;

pub use I32OrStr as Magic;

#[macro_export]
macro_rules! int {
    () => {
        $crate::I32OrStr
    }
}

#[macro_export]
macro_rules! x {
    () => {
        X
    }
}
