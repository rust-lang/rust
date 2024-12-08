#![warn(clippy::missing_errors_doc)]
#![allow(clippy::result_unit_err)]
#![allow(clippy::unnecessary_wraps)]

use std::io;

pub fn pub_fn_missing_errors_header() -> Result<(), ()> {
    //~^ ERROR: docs for function returning `Result` missing `# Errors` section
    //~| NOTE: `-D clippy::missing-errors-doc` implied by `-D warnings`
    unimplemented!();
}

pub async fn async_pub_fn_missing_errors_header() -> Result<(), ()> {
    //~^ ERROR: docs for function returning `Result` missing `# Errors` section
    unimplemented!();
}

/// This is not sufficiently documented.
pub fn pub_fn_returning_io_result() -> io::Result<()> {
    //~^ ERROR: docs for function returning `Result` missing `# Errors` section
    unimplemented!();
}

/// This is not sufficiently documented.
pub async fn async_pub_fn_returning_io_result() -> io::Result<()> {
    //~^ ERROR: docs for function returning `Result` missing `# Errors` section
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
        //~^ ERROR: docs for function returning `Result` missing `# Errors` section
        unimplemented!();
    }

    /// This is not sufficiently documented.
    pub async fn async_pub_method_missing_errors_header() -> Result<(), ()> {
        //~^ ERROR: docs for function returning `Result` missing `# Errors` section
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

    /**
    # Errors
    A description of the errors goes here.
    */
    fn block_comment() -> Result<(), ()> {
        unimplemented!();
    }

    /**
     * # Errors
     * A description of the errors goes here.
     */
    fn block_comment_leading_asterisks() -> Result<(), ()> {
        unimplemented!();
    }

    #[doc(hidden)]
    fn doc_hidden() -> Result<(), ()> {
        unimplemented!();
    }
}

pub trait Trait1 {
    /// This is not sufficiently documented.
    fn trait_method_missing_errors_header() -> Result<(), ()>;
    //~^ ERROR: docs for function returning `Result` missing `# Errors` section

    /// # Errors
    /// A description of the errors goes here.
    fn trait_method_with_errors_header() -> Result<(), ()>;

    #[doc(hidden)]
    fn doc_hidden() -> Result<(), ()> {
        unimplemented!();
    }
}

impl Trait1 for Struct1 {
    fn trait_method_missing_errors_header() -> Result<(), ()> {
        unimplemented!();
    }

    fn trait_method_with_errors_header() -> Result<(), ()> {
        unimplemented!();
    }
}

#[doc(hidden)]
pub trait DocHidden {
    fn f() -> Result<(), ()>;
}

fn main() -> Result<(), ()> {
    Ok(())
}
