- Feature Name: Assert Not Equals Macro (`assert_ne`)
- Start Date: (2016-06-17)
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

`assert_ne` is a macro that takes 2 arguments and panics if they are equal. It
works and is implemented identically to `assert_eq` and serves as its compliment.

# Motivation
[motivation]: #motivation

This feature, among other reasons, makes testing more readable and consistent as
it compliments `asset_eq`. It gives the same style panic message as `assert_eq`,
which eliminates the need to write it yourself.

# Detailed design
[design]: #detailed-design

This feature has exactly the same design and implementation as `assert_eq`.

Here is the definition:

```rust
macro_rules! assert_ne {
    ($left:expr , $right:expr) => ({
        match (&$left, &$right) {
            (left_val, right_val) => {
                if *left_val == *right_val {
                    panic!("assertion failed: `(left !== right)` \
                           (left: `{:?}`, right: `{:?}`)", left_val, right_val)
                }
            }
        }
    })
}
```

# Drawbacks
[drawbacks]: #drawbacks

Any addition to the standard library will need to be maintained forever, so it is
worth weighing the maintenance cost of this over the value add. Given that it is so
similar to `assert_eq`, I believe the weight of this drawback is low.

# Alternatives
[alternatives]: #alternatives

Alternatively, users implement this feature themselves, or use the crate `assert_ne`
that I published.

# Unresolved questions
[unresolved]: #unresolved-questions

None at this moment.
