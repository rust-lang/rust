# `deref_patterns`

The tracking issue for this feature is: [#87121]

[#87121]: https://github.com/rust-lang/rust/issues/87121

------------------------

> **Note**: This feature is incomplete. In the future, it is meant to supersede
> [`box_patterns`] and [`string_deref_patterns`].

This feature permits pattern matching on [smart pointers in the standard library] through their
`Deref` target types, either implicitly or with explicit `deref!(_)` patterns (the syntax of which
is currently a placeholder).

```rust
#![feature(deref_patterns)]
#![allow(incomplete_features)]

let mut v = vec![Box::new(Some(0))];

// Implicit dereferences are inserted when a pattern can match against the
// result of repeatedly dereferencing but can't match against a smart
// pointer itself. This works alongside match ergonomics for references.
if let [Some(x)] = &mut v {
    *x += 1;
}

// Explicit `deref!(_)` patterns may instead be used when finer control is
// needed, e.g. to dereference only a single smart pointer, or to bind the
// the result of dereferencing to a variable.
if let deref!([deref!(opt_x @ Some(1))]) = &mut v {
    opt_x.as_mut().map(|x| *x += 1);
}

assert_eq!(v, [Box::new(Some(2))]);
```

Without this feature, it may be necessary to introduce temporaries to represent dereferenced places
when matching on nested structures:

```rust
let mut v = vec![Box::new(Some(0))];
if let [b] = &mut *v {
    if let Some(x) = &mut **b {
        *x += 1;
    }
}
if let [b] = &mut *v {
    if let opt_x @ Some(1) = &mut **b {
        opt_x.as_mut().map(|x| *x += 1);
    }
}
assert_eq!(v, [Box::new(Some(2))]);
```

Like [`box_patterns`], deref patterns may move out of boxes:

```rust
# #![feature(deref_patterns)]
# #![allow(incomplete_features)]
struct NoCopy;
let deref!(x) = Box::new(NoCopy);
drop::<NoCopy>(x);
```

Additionally, `deref_patterns` implements changes to string and byte string literal patterns,
allowing then to be used in deref patterns:

```rust
# #![feature(deref_patterns)]
# #![allow(incomplete_features)]
match ("test".to_string(), Box::from("test"), b"test".to_vec()) {
    ("test", "test", b"test") => {}
    _ => panic!(),
}

// This works through multiple layers of reference and smart pointer:
match (&Box::new(&"test".to_string()), &&&"test") {
    ("test", "test") => {}
    _ => panic!(),
}

// `deref!("...")` syntax may also be used:
match "test".to_string() {
    deref!("test") => {}
    _ => panic!(),
}

// Matching on slices and arrays using literals is possible elsewhere as well:
match *"test" {
    "test" => {}
    _ => panic!(),
}
match *b"test" {
    b"test" => {}
    _ => panic!(),
}
match *(b"test" as &[u8]) {
    b"test" => {}
    _ => panic!(),
}
```

[`box_patterns`]: ./box-patterns.md
[`string_deref_patterns`]: ./string-deref-patterns.md
[smart pointers in the standard library]: https://doc.rust-lang.org/std/ops/trait.DerefPure.html#implementors
