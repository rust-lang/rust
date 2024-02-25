#![allow(unused)]
#![warn(clippy::empty_docs)]

/// this is a struct
struct Bananas {
    /// count
    count: usize,
}

///
enum Warn {
    ///
    A,
    ///
    B,
}

enum WarnForB {
    /// it's ok
    A,
    ///
    B,
}

#[doc = ""]
#[doc = ""]
fn warn_about_this() {}

#[doc = "a fine function"]
fn this_is_fine() {}

fn warn_about_this_as_well() {
    //!
}

fn this_is_ok() {
    //!
    //! inside the function
}

fn warn() {
    /*! inside the function */
}
