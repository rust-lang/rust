# `macro_metavar_expr_concat`

The tracking issue for this feature is: [#124225]

------------------------

In stable Rust, there is no way to create new identifiers by joining identifiers to literals or other identifiers without using procedural macros such as [`paste`].
 `#![feature(macro_metavar_expr_concat)]` introduces a way to do this, using the concat metavariable expression.

> This feature uses the syntax from [`macro_metavar_expr`] but is otherwise
> independent. It replaces the old unstable feature [`concat_idents`].

> This is an experimental feature; it and its syntax will require a RFC before stabilization.


### Overview

`#![feature(macro_metavar_expr_concat)]` provides the `concat` metavariable expression for creating new identifiers:

```rust
#![feature(macro_metavar_expr_concat)]

macro_rules! create_some_structs {
    ($name:ident) => {
        pub struct ${ concat(First, $name) };
        pub struct ${ concat(Second, $name) };
        pub struct ${ concat(Third, $name) };
    }
}

create_some_structs!(Thing);
```

This macro invocation expands to:

```rust
pub struct FirstThing;
pub struct SecondThing;
pub struct ThirdThing;
```

### Syntax

This feature builds upon the metavariable expression syntax `${ .. }` as specified in [RFC 3086] ([`macro_metavar_expr`]).
 `concat` is available like `${ concat(items) }`, where `items` is a comma separated sequence of idents and/or literals.

### Examples

#### Create a function or method with a concatenated name

```rust
#![feature(macro_metavar_expr_concat)]

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

[`paste`]: https://crates.io/crates/paste
[RFC 3086]: https://rust-lang.github.io/rfcs/3086-macro-metavar-expr.html
[`concat_idents!`]: https://doc.rust-lang.org/nightly/std/macro.concat_idents.html
[`macro_metavar_expr`]: ../language-features/macro-metavar-expr.md
[`concat_idents`]: ../library-features/concat-idents.md
[#124225]: https://github.com/rust-lang/rust/issues/124225
[declarative macros]: https://doc.rust-lang.org/stable/reference/macros-by-example.html
