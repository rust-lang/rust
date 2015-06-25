// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Indicate the state of the build process.
//!
//! `BuildState<T>` wraps around the `Result<T,E>` type and is used
//! to indicate the state of the build process for functions that may
//! fail at runtime. Functions can continue the build by returning
//! `continue_build()` or fail by returning one of the fail states.
//! Use the `try!` macro to chain functions that return `BuildState`.

pub enum ExitStatus {
    SuccessStop,        // Build is successful
    MsgStop(String),    // Build is terminated with a (non-error) message
    ErrStop(String)     // Build is stopped due to an error
}

pub type BuildState<T> = Result<T, ExitStatus>;

impl<T : ::std::fmt::Display> From<T> for ExitStatus {
    fn from(e : T) -> ExitStatus {
        ExitStatus::ErrStop(format!("{}", e))
    }
}

pub fn continue_build() -> BuildState<()> {
    Ok(())
}

pub fn continue_with<T>(v : T) -> BuildState<T> {
    Ok(v)
}

pub fn success_stop<V>() -> BuildState<V> {
    Err(ExitStatus::SuccessStop)
}

pub fn msg_stop<S : Into<String>, V>(s : S) -> BuildState<V> {
    Err(ExitStatus::MsgStop(s.into()))
}

pub fn err_stop<S : Into<String>, V>(s : S) -> BuildState<V> {
    Err(ExitStatus::ErrStop(s.into()))
}

macro_rules! err_stop {
    ( $( $x:expr ),* ) => { return err_stop(format!( $( $x ),* )) }
}
