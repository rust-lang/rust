// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Guarantee that a piece of code is always run.

#[macro_escape];

use ops::Drop;

#[macro_export]
macro_rules! finally {
    ($e: expr) => {
        // this `let` has to be free-standing, so that it's directly
        // in the finally of the callee of `finally!`, to get the dtor to
        // run at the right time.
        let _guard = ::std::finally::FinallyGuard::new(|| $e);
    }
}

/// Runs a user-provided function on destruction.
///
/// One can use this to guarantee that a piece of code is always run,
/// even when the task fails. The `finally!` macro provides a convenient
/// interface, by taking an expression that is wrapped into the
/// appropriate closure.
///
/// # Example
///
/// ```rust
/// {
///     finally!(println!("bye"));
///
///     println!("hi");
/// } // `hi` then `bye`
///
/// {
///     finally!(println!("bye"));
///
///     fail!("oops");
/// } // always prints `bye`
/// ```
pub struct FinallyGuard<'a> {
    priv f: 'a ||
}
impl<'a> FinallyGuard<'a> {
    /// Create a new `FinallyGuard`.
    pub fn new(f: 'a ||) -> FinallyGuard<'a> {
        FinallyGuard { f: f }
    }
}

#[unsafe_destructor]
impl<'a> Drop for FinallyGuard<'a> {
    fn drop(&mut self) {
        (self.f)()
    }
}

#[cfg(test)]
mod test {
    use finally::FinallyGuard;
    use comm::Chan;
    use task::task;

    #[test]
    fn test_no_fail() {
        let mut ran = false;
        {
            finally!(ran = true);
            assert!(!ran);
        }
        assert!(ran)
    }


    #[test]
    fn test_fail() {
        let mut t = task();
        let completion = t.future_result();

        let (p, c) = Chan::new();

        t.spawn(proc() {
                finally!(c.send("guarded"));
                c.send("unguarded");
                fail!()
            });

        // wait for the task to complete
        completion.recv();

        assert_eq!(p.recv(), "unguarded");
        assert_eq!(p.recv(), "guarded");
    }
}
