# `macro_metavar_expr_concat`

The tracking issue for this feature is: [#124225]

------------------------


`#![feature(macro_metavar_expr_concat)]` provides a more powerful alternative to [`concat_idents!`].

> This feature is not to be confused with [`macro_metavar_expr`] or [`concat_idents`].

> This is an experimental feature; it and its syntax will require a RFC before stabilization.


### Overview

`macro_rules!` macros cannot create new identifiers and use them in ident positions.
A common use case is the need to create new structs or functions. The following cannot be done[^1]:

```rust,compile_fail
macro_rules! create_some_structs {
  ($name:ident) => {
      // Invalid syntax
      struct First_$name;
       // Also invalid syntax
      struct Second_($name);
      // Macros are not allowed in this position
      // (This restriction is what makes `concat_idents!` useless)
      struct concat_idents!(Third_, $name);
  }
}
# create_some_structs!(Thing);
```

`#![feature(macro_metavar_expr_concat)]` provides the `concat` metavariable to concatenate idents in ident position:

```rust
#![feature(macro_metavar_expr_concat)]
# #![allow(non_camel_case_types, dead_code)]

macro_rules! create_some_structs {
  ($name:ident) => {
      struct ${ concat(First_, $name) };
      struct ${ concat(Second_, $name) };
      struct ${ concat(Third_, $name) };
  }
}

create_some_structs!(Thing);
```

This macro invocation expands to:

```rust
# #![allow(non_camel_case_types, dead_code)]
struct First_Thing;
struct Second_Thing;
struct Third_Thing;
```

### Syntax

This feature builds upon the metavariable expression syntax `${ .. }` as specified in [RFC 3086] ([`macro_metavar_expr`]).
 `concat` is available like `${ concat(items) }`, where `items` is a comma separated sequence of idents and/or string literals.

### Examples

#### Create a function or method with a concatenated name

```rust
#![feature(macro_metavar_expr_concat)]
# #![allow(non_camel_case_types, dead_code)]

macro_rules! make_getter {
  ($name:ident, $field: ident, $ret:ty) => {
      impl $name {
          pub fn ${ concat(get_, $field) }(&self) -> &$ret {
              &self.$field
          }
      }
  }
}

pub struct Thing {
    description: String,
}

make_getter!(Thing, description, String);
```

This expands to:

```rust
pub struct Thing {
    description: String,
}

impl Thing {
    pub fn get_description(&self) -> &String {
        &self.description
    }
}
```

#### Create names for macro generated tests

```rust
#![feature(macro_metavar_expr_concat)]

macro_rules! test_math {
    ($integer:ident) => {
        #[test]
        fn ${ concat(test_, $integer, _, addition) } () {
            let a: $integer = 73;
            let b: $integer = 42;
            assert_eq!(a + b, 115)
        }

        #[test]
        fn ${ concat(test_, $integer, _, subtraction) } () {
            let a: $integer = 73;
            let b: $integer = 42;
            assert_eq!(a - b, 31)
        }
    }
}

test_math!(i32);
test_math!(u64);
test_math!(u128);
```

Running this returns the following output:

```text
running 6 tests
test test_i32_subtraction ... ok
test test_i32_addition ... ok
test test_u128_addition ... ok
test test_u128_subtraction ... ok
test test_u64_addition ... ok
test test_u64_subtraction ... ok

test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

[^1]: An alternative is the [`paste`] crate.

[`paste`]: https://crates.io/crates/paste
[RFC 3086]: https://rust-lang.github.io/rfcs/3086-macro-metavar-expr.html
[`concat_idents!`]: https://doc.rust-lang.org/nightly/std/macro.concat_idents.html
[`macro_metavar_expr`]: ../language-features/macro-metavar-expr.md
[`concat_idents`]: ../library-features/concat-idents.md
[#124225]: https://github.com/rust-lang/rust/issues/124225
