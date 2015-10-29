- Feature Name: osstring_simple_functions
- Start Date: 2015-10-04
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

Add some additional utility methods to OsString and OsStr.

# Motivation

OsString and OsStr are extremely bare at the moment; some utilities would make them
easier to work with. The given set of utilities is taken from String, and don't add
any additional restrictions to the implementation.

I don't think any of the proposed methods are controversial.

# Detailed design

Add the following methods to OsString:

```rust
/// Creates a new `OsString` with the given capacity. The string will be able
/// to hold exactly `capacity` bytes without reallocating. If `capacity` is 0,
/// the string will not allocate.
///
/// See main `OsString` documentation information about encoding.
fn with_capacity(capacity: usize) -> OsString;

/// Truncates `self` to zero length.
fn clear(&mut self);

/// Returns the number of bytes this `OsString` can hold without reallocating.
///
/// See `OsString` introduction for information about encoding.
fn capacity(&self) -> usize;

/// Reserves capacity for at least `additional` more bytes to be inserted in the
/// given `OsString`. The collection may reserve more space to avoid frequent
/// reallocations.
fn reserve(&mut self, additional: usize);

/// Reserves the minimum capacity for exactly `additional` more bytes to be
/// inserted in the given `OsString`. Does nothing if the capacity is already
/// sufficient.
///
/// Note that the allocator may give the collection more space than it
/// requests. Therefore capacity can not be relied upon to be precisely
/// minimal. Prefer reserve if future insertions are expected.
fn reserve_exact(&mut self, additional: usize);
```

Add the following methods to OsStr:

```rust
/// Checks whether `self` is empty.
fn is_empty(&self) -> bool;

/// Returns the number of bytes in this string.
///
/// See `OsStr` introduction for information about encoding.
fn len(&self) -> usize;
```

# Drawbacks

The meaning of `len()` might be a bit confusing because it's the size of
the internal representation on Windows, which isn't otherwise visible to the
user.

# Alternatives

None.

# Unresolved questions

None.
