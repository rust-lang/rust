// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![unstable(feature = "std_misc")]

/// (Almost) assert_eq!() for floats.
/// Succeeds if `abs(a - b) < 10^-6` or if `abs(a/b - 1) < 10^-6`.
///
/// # Example
///
/// ```
/// assert_approx_eq!(5.0, 5.0);
/// assert_approx_eq!(1000000.0, 1000000.9);
/// assert_approx_eq!(0.0, 0.0000009);
/// ```
///
/// ```should_fail
/// assert_approx_eq!(1.0, 2.0);
/// assert_approx_eq!(1000000.0, 1000001.0);
/// assert_approx_eq!(0.0, 0.000001);
/// ```
#[macro_export]
macro_rules! assert_approx_eq {
    ($a:expr, $b:expr) => ({
        use ::std::num::Float;
        let (a, b) = (&$a, &$b);
        let abs_okay = (*a - *b).abs() < 1.0e-6;
        let rel_okay = (*a / *b - 1.0).abs()  < 1.0e-6;
        assert!(abs_okay || rel_okay, "{} is not approximately equal to {}", *a, *b);
    })
}
