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

## invalid-type-param-default

This lint detects type parameter default erroneously allowed in invalid location. Some
example code that triggers this lint:

```rust,ignore
fn foo<T=i32>(t: T) {}
```

This will produce:

```text
error: defaults for type parameters are only allowed in `struct`, `enum`, `type`, or `trait` definitions.
 --> src/main.rs:4:8
  |
4 | fn foo<T=i32>(t: T) {}
  |        ^
  |
  = note: `#[deny(invalid_type_param_default)]` on by default
  = warning: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
  = note: for more information, see issue #36887 <https://github.com/rust-lang/rust/issues/36887>
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

## patterns-in-fns-without-body

This lint detects patterns in functions without body were that were
previously erroneously allowed. Some example code that triggers this lint:

```rust,compile_fail
trait Trait {
    fn foo(mut arg: u8);
}
```

This will produce:

```text
warning: patterns aren't allowed in methods without bodies
 --> src/main.rs:2:12
  |
2 |     fn foo(mut arg: u8);
  |            ^^^^^^^
  |
  = note: `#[warn(patterns_in_fns_without_body)]` on by default
  = warning: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
  = note: for more information, see issue #35203 <https://github.com/rust-lang/rust/issues/35203>
```

To fix this, remove the pattern; it can be used in the implementation without
being used in the definition. That is:

```rust
trait Trait {
    fn foo(arg: u8);
}

impl Trait for i32 {
    fn foo(mut arg: u8) {

    }
}
```

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

## const-err

This lint detects expressions that will always panic at runtime and would be an
error in a `const` context.

```rust,ignore
let _ = [0; 4][4];
```

This will produce:

```text
error: index out of bounds: the len is 4 but the index is 4
 --> src/lib.rs:1:9
  |
1 | let _ = [0; 4][4];
  |         ^^^^^^^^^
  |
```

## order-dependent-trait-objects

This lint detects a trait coherency violation that would allow creating two
trait impls for the same dynamic trait object involving marker traits.
