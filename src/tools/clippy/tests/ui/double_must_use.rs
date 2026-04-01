#![warn(clippy::double_must_use)]
#![allow(clippy::result_unit_err)]
#![feature(never_type)]

use std::ops::ControlFlow;

#[must_use]
pub fn must_use_result() -> Result<(), ()> {
    //~^ double_must_use

    unimplemented!();
}

#[must_use]
pub fn must_use_tuple() -> (Result<(), ()>, u8) {
    //~^ double_must_use

    unimplemented!();
}

#[must_use]
pub fn must_use_array() -> [Result<(), ()>; 1] {
    //~^ double_must_use

    unimplemented!();
}

#[must_use = "With note"]
pub fn must_use_with_note() -> Result<(), ()> {
    unimplemented!();
}

// vvvv Should not lint (#10486)
#[must_use]
async fn async_must_use() -> usize {
    unimplemented!();
}

#[must_use]
async fn async_must_use_result() -> Result<(), ()> {
    //~^ double_must_use

    Ok(())
}

#[must_use]
pub fn must_use_result_with_uninhabited() -> Result<(), !> {
    unimplemented!();
}

#[must_use]
pub struct T;

#[must_use]
pub fn must_use_result_with_uninhabited_2() -> Result<T, !> {
    //~^ double_must_use
    unimplemented!();
}

#[must_use]
pub fn must_use_controlflow_with_uninhabited() -> ControlFlow<std::convert::Infallible> {
    unimplemented!();
}

#[must_use]
pub fn must_use_controlflow_with_uninhabited_2() -> ControlFlow<std::convert::Infallible, T> {
    //~^ double_must_use
    unimplemented!();
}

fn main() {}
