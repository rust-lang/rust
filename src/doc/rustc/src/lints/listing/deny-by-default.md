# Deny-by-default lints

These lints are all set to the 'deny' level by default.

## exceeding-bitshifts

This lint detects that a shift exceeds the type's number of bits. Some
example code that triggers this lint:

```rust,ignore
1_i32 << 32;
```

This will produce:

```text
error: bitshift exceeds the type's number of bits
 --> src/main.rs:2:5
  |
2 |     1_i32 << 32;
  |     ^^^^^^^^^^^
  |
```

## mutable-transmutes

This lint catches transmuting from `&T` to `&mut T` because it is undefined
behavior. Some example code that triggers this lint:

```rust,ignore
unsafe {
    let y = std::mem::transmute::<&i32, &mut i32>(&5);
}
```

This will produce:

```text
error: mutating transmuted &mut T from &T may cause undefined behavior, consider instead using an UnsafeCell
 --> src/main.rs:3:17
  |
3 |         let y = std::mem::transmute::<&i32, &mut i32>(&5);
  |                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |
```

## no-mangle-const-items

This lint detects any `const` items with the `#[no_mangle]` attribute.
Constants do not have their symbols exported, and therefore, this probably
means you meant to use a `static`, not a `const`. Some example code that
triggers this lint:

```rust,ignore
#[no_mangle]
const FOO: i32 = 5;
```

This will produce:

```text
error: const items should never be `#[no_mangle]`
 --> src/main.rs:3:1
  |
3 | const FOO: i32 = 5;
  | -----^^^^^^^^^^^^^^
  | |
  | help: try a static value: `pub static`
  |
```

## overflowing-literals

This lint detects literal out of range for its type. Some
example code that triggers this lint:

```rust,compile_fail
let x: u8 = 1000;
```

This will produce:

```text
error: literal out of range for u8
 --> src/main.rs:2:17
  |
2 |     let x: u8 = 1000;
  |                 ^^^^
  |
```

## parenthesized-params-in-types-and-modules

This lint detects incorrect parentheses. Some example code that triggers this
lint:

```rust,ignore
let x = 5 as usize();
```

This will produce:

```text
error: parenthesized parameters may only be used with a trait
 --> src/main.rs:2:21
  |
2 |   let x = 5 as usize();
  |                     ^^
  |
  = note: `#[deny(parenthesized_params_in_types_and_modules)]` on by default
  = warning: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
  = note: for more information, see issue #42238 <https://github.com/rust-lang/rust/issues/42238>
```

To fix it, remove the `()`s.

## pub-use-of-private-extern-crate

This lint detects a specific situation of re-exporting a private `extern crate`;

## unknown-crate-types

This lint detects an unknown crate type found in a `#[crate_type]` directive. Some
example code that triggers this lint:

```rust,ignore
#![crate_type="lol"]
```

This will produce:

```text
error: invalid `crate_type` value
 --> src/lib.rs:1:1
  |
1 | #![crate_type="lol"]
  | ^^^^^^^^^^^^^^^^^^^^
  |
```
