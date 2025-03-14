# `deref_patterns`

The tracking issue for this feature is: [#87121]

[#87121]: https://github.com/rust-lang/rust/issues/87121

------------------------

This feature permits pattern matching on [library-defined smart pointers] through their `Deref`
target types, either implicitly or with the placeholder `deref!(_)` syntax.

```rust
#![feature(deref_patterns)]
#![allow(incomplete_features)]

pub fn main() {
    let mut v = vec![Box::new(Some(0))];
    if let [Some(ref mut x)] = v {
        *x += 1;
    }
    if let deref!([deref!(Some(ref mut x))]) = v {
        *x += 1;
    }
    assert_eq!(v, [Box::new(Some(2))]);
}
```

Without this feature, it may be necessary to introduce temporaries to represent dereferenced places
when matching on nested structures:

```rust
pub fn main() {
    let mut v = vec![Box::new(Some(0))];
    if let [ref mut b] = *v {
        if let Some(ref mut x) = **b {
            *x += 1;
        }
    }
    if let [ref mut b] = *v {
        if let Some(ref mut x) = **b {
            *x += 1;
        }
    }
    assert_eq!(v, [Box::new(Some(2))]);
}
```

[library-defined smart pointers]: https://doc.rust-lang.org/std/ops/trait.DerefPure.html#implementors
