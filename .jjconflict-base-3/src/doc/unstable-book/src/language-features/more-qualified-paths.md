# `more_qualified_paths`

The `more_qualified_paths` feature can be used in order to enable the
use of qualified paths in patterns.

The tracking issue for this feature is: [#86935](https://github.com/rust-lang/rust/issues/86935).

------------------------

## Example

```rust
#![feature(more_qualified_paths)]

fn main() {
    // destructure through a qualified path
    let <Foo as A>::Assoc { br } = StructStruct { br: 2 };
}

struct StructStruct {
    br: i8,
}

struct Foo;

trait A {
    type Assoc;
}

impl A for Foo {
    type Assoc = StructStruct;
}
```
