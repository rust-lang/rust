- Feature Name: clamp functions
- Start Date: 2017-03-26
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

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

Likely locations would be in std::cmp::clamp implemented for all Ord types, and a special version implemented for f32 and f64.
The f32 and f64 versions could live either in std::cmp or in the primitive types themselves.  There are good arguments for either
location.

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

Add the following to std::cmp

```Rust
use ops::RangeInclusive;
/// Returns the upper bound of the range if input is greater than the range, and the lower bound of
/// the range if input is less than the range.  Otherwise this will return input.
#[inline]
pub fn clamp<T: Ord>(input: T, range: RangeInclusive<T>) -> T {
    if let RangeInclusive::NonEmpty{start, end} = range {
        if input < start {
            return start;
        }
        else if input > end {
            return end;
        }
        else {
            return input;
        }
    }
    else {
        // This should never be executed.
        return input;
    }
}
```

And the following to libstd/f32.rs, and a similar version for f64

```Rust
use ops::RangeInclusive;
/// Returns the upper bound if self is greater than the bound, and the lower bound if self is less than the bound.
/// Otherwise this returns self.
///
/// # Examples
///
/// ```
/// assert!((-3.0f32).clamp(-2.0f32...1.0f32) == -2.0f32);
/// assert!((0.0f32).clamp(-2.0f32...1.0f32) == 0.0f32);
/// assert!((2.0f32).clamp(-2.0f32...1.0f32) == 1.0f32);
/// ```
#[inline]
pub fn clamp(self, range: RangeInclusive<f32>) -> f32 {
  if let NonEmpty{start, end} = range {
      if self < start {
          return start;
      }
      else if self > max {
          return max;
      }
      else {
          return self;
      }
  }
  else {
      // This should never be executed.
      return NAN;
  }
}
```

There are 3 special float values the clamp function will need to handle, and 3 positions into which they can go so I will represent
the edge case behavior with a 3x3 chart.

|  |INFINITY|NEG_INFINITY|NAN|
|---|---|---|---|
|self|return max;|return min;|return NAN;|
|upper bound|No upper bound enforced|return NEG_INFINITY;|No upper bound enforced|
|lower bound|return INFINITY;|No lower bound enforced|No lower bound enforced|

# How We Teach This
[how-we-teach-this]: #how-we-teach-this

The proposed changes would not mandate modifications to any Rust educational material.

# Drawbacks
[drawbacks]: #drawbacks

This is trivial to implement downstream, and several versions of it exist downstream.

# Alternatives
[alternatives]: #alternatives

Alternatives were explored at https://internals.rust-lang.org/t/clamp-function-for-primitive-types/4999

# Unresolved questions
[unresolved]: #unresolved-questions

Should the float version of the clamp function live in f32 and f64, or in std::cmp as that's where the Ord version would go?
