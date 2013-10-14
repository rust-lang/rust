// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A utility class which executes code when it falls out of scope.
//!
//! The `guard` function returns an object which will conditionally execute the
//! block passed to it when it falls out of scope:
//!
//! ```rust
//! use std::scope;
//! use std::scope::Success;
//!
//! let _guard = do scope::guard(Success) {
//!     println!("I am falling out of scope and the task hasn't failed!");
//! }
//! ```

use task;
use prelude::*;

/// Specifies the condition under which the block will be executed
pub enum ScopeCondition {
    /// Specifies that the block should be executed if the task has not failed.
    Success,
    /// Specifies that the block should be executed if the task has failed.
    Failure,
    /// Specifies that the block should be executed whether or not the task has
    /// failed.
    Exit
}

/// A struct which conditionally executes a block when it falls out of scope.
pub struct ScopeGuard<'self> {
    priv blk: &'self fn(),
    priv method: ScopeCondition
}

#[unsafe_destructor]
impl<'self> Drop for ScopeGuard<'self> {
    fn drop(&mut self) {
        match self.method {
            Success if task::failing() => (),
            Failure if !task::failing() => (),
            _ => (self.blk)()
        }
    }
}

/// Returns an object which will execute the provided block when it falls out
/// of scope if the specified condition is met.
pub fn guard<'a>(method: ScopeCondition, blk: &'a fn()) -> ScopeGuard<'a> {
    ScopeGuard {
        method: method,
        blk: blk
    }
}

#[cfg(test)]
mod test {
    use super::{Success, Failure, Exit, guard};
    use prelude::*;
    use task;

    #[test]
    fn test_success() {
        let mut a = 0;
        {
            let _guard = do guard(Success) { a += 1; };
            assert_eq!(a, 0);
        }
        assert_eq!(a, 1);
    }

    #[test]
    fn test_success_failure() {
        static mut a: int = 0;
        let _: Result<(), ()> = do task::try {
            let _guard = do guard(Success) { unsafe { a += 1; } };
            fail2!();
        };
        assert_eq!(unsafe { a }, 0);
    }

    #[test]
    fn test_exit_success() {
        let mut a = 0;
        {
            let _guard = do guard(Exit) { a += 1; };
            assert_eq!(a, 0);
        }
        assert_eq!(a, 1);
    }

    #[test]
    fn test_exit_failure() {
        static mut a: int = 0;
        let _: Result<(), ()> = do task::try {
            let _guard = do guard(Exit) { unsafe { a += 1; } };
            assert_eq!(unsafe { a }, 0);
            fail2!();
        };
        assert_eq!(unsafe { a }, 1);
    }

    #[test]
    fn test_failure() {
        static mut a: int = 0;
        let _: Result<(), ()> = do task::try {
            let _guard = do guard(Failure) { unsafe { a += 1; } };
            assert_eq!(unsafe { a }, 0);
            fail2!();
        };
        assert_eq!(unsafe { a }, 1);
    }

    #[test]
    fn test_failure_success() {
        let mut a = 0;
        {
            let _guard = do guard(Failure) { a += 1; };
        }
        assert_eq!(a, 0);
    }
}

