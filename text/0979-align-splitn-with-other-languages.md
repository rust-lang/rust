- Feature Name: n/a
- Start Date: 2015-03-15
- RFC PR: https://github.com/rust-lang/rfcs/pull/979
- Rust Issue: https://github.com/rust-lang/rust/issues/23911

# Summary

Make the `count` parameter of `SliceExt::splitn`, `StrExt::splitn` and
corresponding reverse variants mean the *maximum number of items
returned*, instead of the *maximum number of times to match the
separator*.

# Motivation

The majority of other languages (see examples below) treat the `count`
parameter as the maximum number of items to return. Rust already has
many things newcomers need to learn, making other things similar can
help adoption.

# Detailed design

Currently `splitn` uses the `count` parameter to decide how many times
the separator should be matched:

```rust
let v: Vec<_> = "a,b,c".splitn(2, ',').collect();
assert_eq!(v, ["a", "b", "c"]);
```

The simplest change we can make is to decrement the count in the
constructor functions. If the count becomes zero, we mark the returned
iterator as `finished`. See **Unresolved questions** for nicer
transition paths.

## Example usage

### Strings

```rust
let input = "a,b,c";
let v: Vec<_> = input.splitn(2, ',').collect();
assert_eq!(v, ["a", "b,c"]);

let v: Vec<_> = input.splitn(1, ',').collect();
assert_eq!(v, ["a,b,c"]);

let v: Vec<_> = input.splitn(0, ',').collect();
assert_eq!(v, []);
```

### Slices

```rust
let input = [1, 0, 2, 0, 3];
let v: Vec<_> = input.splitn(2, |&x| x == 0).collect();
assert_eq!(v, [[1], [2, 0, 3]]);

let v: Vec<_> = input.splitn(1, |&x| x == 0).collect();
assert_eq!(v, [[1, 0, 2, 0, 3]]);

let v: Vec<_> = input.splitn(0, |&x| x == 0).collect();
assert_eq!(v, []);
```

## Languages where `count` is the maximum number of items returned

### C# ###

```c#
"a,b,c".Split(new char[] {','}, 2)
// ["a", "b,c"]
```

### Clojure

```clojure
(clojure.string/split "a,b,c" #"," 2)
;; ["a" "b,c"]
```

### Go

```go
strings.SplitN("a,b,c", ",", 2)
// [a b,c]
```

### Java

```java
"a,b,c".split(",", 2);
// ["a", "b,c"]
```

### Ruby

```ruby
"a,b,c".split(',', 2)
# ["a", "b,c"]
```

### Perl

```perl
split(",", "a,b,c", 2)
# ['a', 'b,c']
```

## Languages where `count` is the maximum number of times the separator will be matched

### Python

```python
"a,b,c".split(',', 2)
# ['a', 'b', 'c']
```

### Swift

```swift
split("a,b,c", { $0 == "," }, maxSplit: 2)
// ["a", "b", "c"]
```

# Drawbacks

Changing the *meaning* of the `count` parameter without changing the
*type* is sure to cause subtle issues. See **Unresolved questions**.

The iterator can only return 2^64 values; previously we could return
2^64 + 1. This could also be considered an upside, as we can now
return an empty iterator.

# Alternatives

1. Keep the status quo. People migrating from many other languages
will continue to be surprised.

2. Add a parallel set of functions that clearly indicate that `count`
is the maximum number of items that can be returned.

# Unresolved questions

Is there a nicer way to change the behavior of `count` such that users
of `splitn` get compile-time errors when migrating?

1. Add a dummy parameter, and mark the methods unstable. Remove the
parameterand re-mark as stable near the end of the beta period.

2. Move the methods from `SliceExt` and `StrExt` to a new trait that
needs to be manually imported. After the transition, move the methods
back and deprecate the trait. This would not break user code that
migrated to the new semantic.
