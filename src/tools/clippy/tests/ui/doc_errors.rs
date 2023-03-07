#![warn(clippy::missing_errors_doc)]
#![allow(clippy::result_unit_err)]
#![allow(clippy::unnecessary_wraps)]

use std::io;

pub fn pub_fn_missing_errors_header() -> Result<(), ()> {
    unimplemented!();
}

pub async fn async_pub_fn_missing_errors_header() -> Result<(), ()> {
    unimplemented!();
}

/// This is not sufficiently documented.
pub fn pub_fn_returning_io_result() -> io::Result<()> {
    unimplemented!();
}

/// This is not sufficiently documented.
pub async fn async_pub_fn_returning_io_result() -> io::Result<()> {
    unimplemented!();
}

/// # Errors
/// A description of the errors goes here.
pub fn pub_fn_with_errors_header() -> Result<(), ()> {
    unimplemented!();
}

/// # Errors
/// A description of the errors goes here.
pub async fn async_pub_fn_with_errors_header() -> Result<(), ()> {
    unimplemented!();
}

/// This function doesn't require the documentation because it is private
fn priv_fn_missing_errors_header() -> Result<(), ()> {
    unimplemented!();
}

/// This function doesn't require the documentation because it is private
async fn async_priv_fn_missing_errors_header() -> Result<(), ()> {
    unimplemented!();
}

pub struct Struct1;

impl Struct1 {
    /// This is not sufficiently documented.
    pub fn pub_method_missing_errors_header() -> Result<(), ()> {
        unimplemented!();
    }

    /// This is not sufficiently documented.
    pub async fn async_pub_method_missing_errors_header() -> Result<(), ()> {
        unimplemented!();
    }

    /// # Errors
    /// A description of the errors goes here.
    pub fn pub_method_with_errors_header() -> Result<(), ()> {
        unimplemented!();
    }

    /// # Errors
    /// A description of the errors goes here.
    pub async fn async_pub_method_with_errors_header() -> Result<(), ()> {
        unimplemented!();
    }

    /// This function doesn't require the documentation because it is private.
    fn priv_method_missing_errors_header() -> Result<(), ()> {
        unimplemented!();
    }

    /// This function doesn't require the documentation because it is private.
    async fn async_priv_method_missing_errors_header() -> Result<(), ()> {
        unimplemented!();
    }
}

pub trait Trait1 {
    /// This is not sufficiently documented.
    fn trait_method_missing_errors_header() -> Result<(), ()>;

    /// # Errors
    /// A description of the errors goes here.
    fn trait_method_with_errors_header() -> Result<(), ()>;
}

impl Trait1 for Struct1 {
    fn trait_method_missing_errors_header() -> Result<(), ()> {
        unimplemented!();
    }

    fn trait_method_with_errors_header() -> Result<(), ()> {
        unimplemented!();
    }
}

fn main() -> Result<(), ()> {
    Ok(())
}
