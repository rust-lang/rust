# `catch_expr`

The tracking issue for this feature is: [#31436]

[#31436]: https://github.com/rust-lang/rust/issues/31436

------------------------

The `catch_expr` feature adds support for a `catch` expression. The `catch`
expression creates a new scope one can use the `?` operator in.

```rust
#![feature(catch_expr)]

use std::num::ParseIntError;

let result: Result<i32, ParseIntError> = do catch {
    "1".parse::<i32>()?
        + "2".parse::<i32>()?
        + "3".parse::<i32>()?
};
assert_eq!(result, Ok(6));

let result: Result<i32, ParseIntError> = do catch {
    "1".parse::<i32>()?
        + "foo".parse::<i32>()?
        + "3".parse::<i32>()?
};
assert!(result.is_err());
```
