/* SPDX-License-Identifier: MIT OR Apache-2.0 */
//! IEEE 754-2008 `minNum`. This has been superseded by IEEE 754-2019 `minimumNumber`.
//!
//! Per the spec, returns the canonicalized result of:
//! - `x` if `x < y`
//! - `y` if `y < x`
//! - The other number if one is NaN
//! - Otherwise, either `x` or `y`, canonicalized
//! - -0.0 and +0.0 may be disregarded (unlike newer operations)
//!
//! We do not treat sNaN and qNaN differently; even though IEEE technically requires this, (a call
//! with sNaN should return qNaN rather than the other result), it breaks associativity so isn't
//! desired behavior. C23 does not differentiate between sNaN and qNaN, so we do not either. More
//! on the problems with `minNum` [here][minnum-removal].
//!
//! IEEE also specifies that a sNaN in either argument should signal invalid, but we do not
//! implement this.
//!
//! [minnum-removal]: https://grouper.ieee.org/groups/msc/ANSI_IEEE-Std-754-2019/background/minNum_maxNum_Removal_Demotion_v3.pdf

use crate::support::Float;

#[inline]
pub fn fmin<F: Float>(x: F, y: F) -> F {
    let res = if y.is_nan() || x < y { x } else { y };
    res.canonicalize()
}
