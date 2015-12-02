% `type` Aliases

The `type` keyword lets you declare an alias of another type:

```rust
type Name = String;
```

You can then use this type as if it were a real type:

```rust
type Name = String;

let x: Name = "Hello".to_string();
```

Note, however, that this is an _alias_, not a new type entirely. In other
words, because Rust is strongly typed, you’d expect a comparison between two
different types to fail:

```rust,ignore
let x: i32 = 5;
let y: i64 = 5;

if x == y {
   // ...
}
```

this gives

```text
error: mismatched types:
 expected `i32`,
    found `i64`
(expected i32,
    found i64) [E0308]
     if x == y {
             ^
```

But, if we had an alias:

```rust
type Num = i32;

let x: i32 = 5;
let y: Num = 5;

if x == y {
   // ...
}
```

This compiles without error. Values of a `Num` type are the same as a value of
type `i32`, in every way. You can use [tuple struct] to really get a new type.

[tuple struct]: structs.html#tuple-structs

You can also use type aliases with generics:

```rust
use std::result;

enum ConcreteError {
    Foo,
    Bar,
}

type Result<T> = result::Result<T, ConcreteError>;
```

This creates a specialized version of the `Result` type, which always has a
`ConcreteError` for the `E` part of `Result<T, E>`. This is commonly used
in the standard library to create custom errors for each subsection. For
example, [io::Result][ioresult].

[ioresult]: ../std/io/type.Result.html
