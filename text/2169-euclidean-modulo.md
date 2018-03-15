- Feature Name: `euclidean_modulo`
- Start Date: 2017-10-09
- RFC PR: https://github.com/rust-lang/rfcs/pull/2169
- Rust Issue: https://github.com/rust-lang/rust/issues/49048

# Summary
[summary]: #summary

This RFC proposes the addition of a modulo method with more useful and mathematically regular properties over the built-in remainder `%` operator when the dividend or divisor is negative, along with the associated division method.

For previous discussion, see: https://internals.rust-lang.org/t/mathematical-modulo-operator/5952.

# Motivation
[motivation]: #motivation

The behaviour of division and modulo, as implemented by Rust's (truncated) division `/` and remainder (or truncated modulo) `%` operators, with respect to negative operands is unintuitive and has fewer useful mathematical properties than that of other varieties of division and modulo, such as flooring and Euclidean[[1]](https://dl.acm.org/citation.cfm?doid=128861.128862). While there are good reasons for this design decision[[2]](https://mail.mozilla.org/pipermail/rust-dev/2013-April/003786.html), having convenient access to a modulo operation, in addition to the remainder is very useful, and has often been requested[[3]](https://mail.mozilla.org/pipermail/rust-dev/2013-April/003680.html)[[4]](https://github.com/rust-lang/rust/issues/13909)[[5]](https://stackoverflow.com/questions/31210357/is-there-a-modulus-not-remainder-function-operation)[[6]](https://users.rust-lang.org/t/proper-modulo-support/903)[[7]](https://www.reddit.com/r/rust/comments/3yoo1q/remainder/).

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

```rust
// Comparison of the behaviour of Rust's truncating division
// and remainder, vs Euclidean division & modulo.
(-8 / 3,      -8 % 3)           // (-2, -2)
((-8).div_euc(3), (-8).mod_euc(3))  // (-3,  1)
```
Euclidean division & modulo for integers and floating-point numbers will be achieved using the `div_euc` and `mod_euc` methods. The `%` operator has identical behaviour to `mod_euc` for unsigned integers. However, when using signed integers or floating-point numbers, you should be careful to consider the behaviour you want: often Euclidean modulo will be more appropriate.

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

It is important to have both division and modulo methods, as the two operations are intrinsically linked[[8]](https://en.wikipedia.org/wiki/Modulo_operation), though it is often the modulo operator that is specifically requested.

A complete implementation of Euclidean modulo would involve adding 8 methods to the integer primitives in `libcore/num/mod.rs` and 2 methods to the floating-point primitives in `libcore/num` and `libstd`:
```rust
// Implemented for all numeric primitives.
fn div_euc(self, rhs: Self) -> Self;

fn mod_euc(self, rhs: Self) -> Self;

// Implemented for all integer primitives (signed and unsigned).
fn checked_div_euc(self, other: Self) -> Option<Self>;
fn overflowing_div_euc(self, rhs: Self) -> (Self, bool);
fn wrapping_div_euc(self, rhs: Self) -> Self;

fn checked_mod_euc(self, other: Self) -> Option<Self>;
fn overflowing_mod_euc(self, rhs: Self) -> (Self, bool);
fn wrapping_mod_euc(self, rhs: Self) -> Self;
```

Sample implementations for `div_euc` and `mod_euc` on signed integers:
```rust
fn div_euc(self, rhs: Self) -> Self {
    let q = self / rhs;
    if self % rhs < 0 {
        return if rhs > 0 { q - 1 } else { q + 1 }
    }
    q
}

fn mod_euc(self, rhs: Self) -> Self {
    let r = self % rhs;
    if r < 0 {
        return if rhs > 0 { r + rhs } else { r - rhs }
    }
    r
}
```
And on `f64` (analagous to the `f32` implementation):
```rust
fn div_euc(self, rhs: f64) -> f64 {
    let q = (self / rhs).trunc();
    if self % rhs < 0.0 {
        return if rhs > 0.0 { q - 1.0 } else { q + 1.0 }
    }
    q
}

fn mod_euc(self, rhs: f64) -> f64 {
    let r = self % rhs;
    if r < 0.0 {
        return if rhs > 0.0 { r + rhs } else { r - rhs }
    }
    r
}
```

The unsigned implementations of these methods are trivial.
The `checked_*`, `overflowing_*` and `wrapping_*` methods would operate analogously to their non-Euclidean `*_div` and `*_rem` counterparts that already exist. The edge cases are identical.

# Drawbacks
[drawbacks]: #drawbacks

Standard drawbacks of adding methods to primitives apply. However, with the proposed method names, there are unlikely to be conflicts downstream[[9]](https://github.com/search?q=div_euc+language%3ARust&type=Code&utf8=%E2%9C%93)[[10]](https://github.com/search?q=mod_euc+language%3ARust&type=Code&utf8=%E2%9C%93).

# Rationale and alternatives
[alternatives]: #alternatives

Flooring modulo is another variant that also has more useful behaviour with negative dividends than the remainder (truncating modulo). The difference in behaviour between flooring and Euclidean division & modulo come up rarely in practice, but there are arguments in favour of the mathematical properties of Euclidean division and modulo[[1]](https://dl.acm.org/citation.cfm?doid=128861.128862). Alternatively, both methods (flooring _and_ Euclidean) could be made available, though the difference between the two is likely specialised-enough that this would be overkill.

The functionality could be provided as an operator. However, it is likely that the functionality of remainder and modulo are small enough that it is not worth providing a dedicated operator for the method.

This functionality could instead reside in a separate crate, such as `num` (floored division & modulo is already available in this crate). However, there are strong points for inclusion into core itself:
- Modulo as an operation is more often desirable than remainder for signed operations (so much so that it is the default in a number of languages) -- [the mailing list discussion has more support in favour of flooring/Euclidean division](https://mail.mozilla.org/pipermail/rust-dev/2013-April/003687.html).
- Many people are unaware that the remainder can cause problems with signed integers, and having a method displaying the other behaviour would draw attention to this subtlety.
- The previous support for this functionality in core shows that many are keen to have this available.
- The Euclidean or flooring modulo is used (or reimplemented) commonly enough that it is worth having it generally accessible, rather than in a separate crate that must be depended on by each project.

# Unresolved questions
[unresolved]: #unresolved-questions

None.
