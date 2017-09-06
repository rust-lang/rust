- Feature Name: clamp functions
- Start Date: 2017-03-26
- RFC PR: https://github.com/rust-lang/rfcs/pull/1961/
- Rust Issue: https://github.com/rust-lang/rust/issues/44095

# Summary
[summary]: #summary

Add functions to the language which take a value and an inclusive range, and will "clamp" the input to the range.  I.E.

```Rust
if input > max {
    return max;
}
else if input < min {
    return min;
} else {
    return input;
}
```

These would be on the Ord trait, and have a special version implemented for f32 and f64.

# Motivation
[motivation]: #motivation

Clamp is a very common pattern in Rust libraries downstream.  Some observed implementations of this include:

http://nalgebra.org/rustdoc/nalgebra/fn.clamp.html

http://rust-num.github.io/num/num/fn.clamp.html

Many libraries don't expose or consume a clamp function but will instead use patterns like this:
```Rust
if input > max {
    max
}
else if input < min {
    min
} else {
    input
}
```
and
```Rust
input.max(min).min(max);
```
and even
```Rust
match input {
    c if c >  max =>  max,
    c if c < min => min,
    c              =>  c,
}
```

Typically these patterns exist where there is a need to interface with APIs that take normalized values or when sending 
output to hardware that expects values to be in a certain range, such as audio samples or painting to pixels on a display.

While this is pretty trivial to implement downstream there are quite a few ways to do it and just writing the clamp 
inline usually results in rather a lot of control flow structure to describe a fairly simple and common concept.

# Detailed design
[design]: #detailed-design

Add the following to std::cmp::Ord

```Rust
/// Returns max if self is greater than max, and min if self is less than min.  
/// Otherwise this will return self.  Panics if min > max.
#[inline]
pub fn clamp(self, min: Self, max: Self) -> Self {
    assert!(min <= max);
    if self < min {
        min
    }
    else if self > max {
        max
    } else {
        self
    }
}
```

And the following to libstd/f32.rs, and a similar version for f64

```Rust
/// Returns max if self is greater than max, and min if self is less than min.
/// Otherwise this returns self.  Panics if min > max, min equals NaN, or max equals NaN.
///
/// # Examples
///
/// ```
/// assert!((-3.0f32).clamp(-2.0f32, 1.0f32) == -2.0f32);
/// assert!((0.0f32).clamp(-2.0f32, 1.0f32) == 0.0f32);
/// assert!((2.0f32).clamp(-2.0f32, 1.0f32) == 1.0f32);
/// ```
pub fn clamp(self, min: f32, max: f32) -> f32 {
    assert!(min <= max);
    let mut x = self;
    if x < min { x = min; }
    if x > max { x = max; }
    x
}
```

This NaN handling behavior was chosen because a range with NaN on either side isn't really a range at all and the function can't be guaranteed to behave correctly if that is the case.

# How We Teach This
[how-we-teach-this]: #how-we-teach-this

The proposed changes would not mandate modifications to any Rust educational material.

# Drawbacks
[drawbacks]: #drawbacks

This is trivial to implement downstream, and several versions of it exist downstream.

# Alternatives
[alternatives]: #alternatives

Alternatives were explored at https://internals.rust-lang.org/t/clamp-function-for-primitive-types/4999

Additionally there is the option of placing clamp in std::cmp in order to avoid backwards compatibility problems.  This is however semantically undesirable, as `1.clamp(2, 3);` is more readable than `clamp(1, 2, 3);`

# Unresolved questions
[unresolved]: #unresolved-questions

None
