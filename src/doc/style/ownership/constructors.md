% Constructors

### Define constructors as static, inherent methods. [FIXME: needs RFC]

In Rust, "constructors" are just a convention:

```rust
impl<T> Vec<T> {
    pub fn new() -> Vec<T> { ... }
}
```

Constructors are static (no `self`) inherent methods for the type that they
construct. Combined with the practice of
[fully importing type names](../style/imports.md), this convention leads to
informative but concise construction:

```rust
use vec::Vec;

// construct a new vector
let mut v = Vec::new();
```

This convention also applied to conversion constructors (prefix `from` rather
than `new`).

### Provide constructors for passive `struct`s with defaults. [FIXME: needs RFC]

Given the `struct`

```rust
pub struct Config {
    pub color: Color,
    pub size:  Size,
    pub shape: Shape,
}
```

provide a constructor if there are sensible defaults:

```rust
impl Config {
    pub fn new() -> Config {
        Config {
            color: Brown,
            size: Medium,
            shape: Square,
        }
    }
}
```

which then allows clients to concisely override using `struct` update syntax:

```rust
Config { color: Red, .. Config::new() };
```

See the [guideline for field privacy](../features/types/README.md) for
discussion on when to create such "passive" `struct`s with public
fields.
