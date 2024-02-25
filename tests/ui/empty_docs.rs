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

enum WarnA {
    ///
    A,
    B,
}

enum DontWarn {
    /// it's ok
    A,
    ///
    B,
}

#[doc = ""]
fn warn_about_this() {}

#[doc = ""]
#[doc = ""]
fn this_doesn_warn() {}

#[doc = "a fine function"]
fn this_is_fine() {}

fn warn_about_this_as_well() {
    //!
}

///
fn warn_inner_outer() {
    //!w
}

fn this_is_ok() {
    //!
    //! inside the function
}

fn warn() {
    /*! */
}

fn dont_warn() {
    /*! dont warn me */
}
