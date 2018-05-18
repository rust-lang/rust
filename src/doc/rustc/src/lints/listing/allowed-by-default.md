# Allowed-by-default lints

These lints are all set to the 'allow' level by default. As such, they won't show up
unless you set them to a higher lint level with a flag or attribute.

## anonymous-parameters

This lint detects anonymous parameters. Some example code that triggers this lint:

```rust
trait Foo {
    fn foo(usize);
}
```

When set to 'deny', this will produce:

```text
error: use of deprecated anonymous parameter
 --> src/lib.rs:5:11
  |
5 |     fn foo(usize);
  |           ^
  |
  = warning: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
  = note: for more information, see issue #41686 <https://github.com/rust-lang/rust/issues/41686>
```

This syntax is mostly a historical accident, and can be worked around quite
easily:

```rust
trait Foo {
    fn foo(_: usize);
}
```

## bare-trait-object

This lint suggests using `dyn Trait` for trait objects. Some example code
that triggers this lint:

```rust
#![feature(dyn_trait)]

trait Trait { }

fn takes_trait_object(_: Box<Trait>) {
}
```

When set to 'deny', this will produce:

```text
error: trait objects without an explicit `dyn` are deprecated
 --> src/lib.rs:7:30
  |
7 | fn takes_trait_object(_: Box<Trait>) {
  |                              ^^^^^ help: use `dyn`: `dyn Trait`
  |
```

To fix it, do as the help message suggests:

```rust
#![feature(dyn_trait)]
#![deny(bare_trait_objects)]

trait Trait { }

fn takes_trait_object(_: Box<dyn Trait>) {
}
```

## box-pointers

This lints use of the Box type. Some example code that triggers this lint:

```rust
struct Foo {
    x: Box<isize>,
}
```

When set to 'deny', this will produce:

```text
error: type uses owned (Box type) pointers: std::boxed::Box<isize>
 --> src/lib.rs:6:5
  |
6 |     x: Box<isize> //~ ERROR type uses owned
  |     ^^^^^^^^^^^^^
  |
```

This lint is mostly historical, and not particularly useful. `Box<T>` used to
be built into the language, and the only way to do heap allocation. Today's
Rust can call into other allocators, etc.

## elided-lifetime-in-path

This lint detects the use of hidden lifetime parameters. Some example code
that triggers this lint:

```rust
struct Foo<'a> {
    x: &'a u32
}

fn foo(x: &Foo) {
}
```

When set to 'deny', this will produce:

```text
error: hidden lifetime parameters are deprecated, try `Foo<'_>`
 --> src/lib.rs:5:12
  |
5 | fn foo(x: &Foo) {
  |            ^^^
  |
```

Lifetime elision elides this lifetime, but that is being deprecated.

## missing-copy-implementations

This lint detects potentially-forgotten implementations of `Copy`. Some
example code that triggers this lint:

```rust
pub struct Foo {
    pub field: i32
}
```

When set to 'deny', this will produce:

```text
error: type could implement `Copy`; consider adding `impl Copy`
 --> src/main.rs:3:1
  |
3 | / pub struct Foo { //~ ERROR type could implement `Copy`; consider adding `impl Copy`
4 | |     pub field: i32
5 | | }
  | |_^
  |
```

You can fix the lint by deriving `Copy`.

This lint is set to 'allow' because this code isn't bad; it's common to write
newtypes like this specifically so that a `Copy` type is no longer `Copy`.

## missing-debug-implementations

This lint detects missing implementations of `fmt::Debug`. Some example code
that triggers this lint:

```rust
pub struct Foo;
```

When set to 'deny', this will produce:

```text
error: type does not implement `fmt::Debug`; consider adding #[derive(Debug)] or a manual implementation
 --> src/main.rs:3:1
  |
3 | pub struct Foo;
  | ^^^^^^^^^^^^^^^
  |
```

You can fix the lint by deriving `Debug`.

## missing-docs

This lint detects missing documentation for public items. Some example code
that triggers this lint:

```rust
pub fn foo() {}
```

When set to 'deny', this will produce:

```text
error: missing documentation for crate
 --> src/main.rs:1:1
  |
1 | / #![deny(missing_docs)]
2 | |
3 | | pub fn foo() {}
4 | |
5 | | fn main() {}
  | |____________^
  |

error: missing documentation for a function
 --> src/main.rs:3:1
  |
3 | pub fn foo() {}
  | ^^^^^^^^^^^^

```

To fix the lint, add documentation to all items.

## single-use-lifetime

This lint detects lifetimes that are only used once. Some example code that
triggers this lint:

```rust
struct Foo<'x> {
    x: &'x u32
}
```

When set to 'deny', this will produce:

```text
error: lifetime name `'x` only used once
 --> src/main.rs:3:12
  |
3 | struct Foo<'x> {
  |            ^^
  |
```

## trivial-casts

This lint detects trivial casts which could be removed. Some example code
that triggers this lint:

```rust
let x: &u32 = &42;
let _ = x as *const u32;
```

When set to 'deny', this will produce:

```text
error: trivial cast: `&u32` as `*const u32`. Cast can be replaced by coercion, this might require type ascription or a temporary variable
 --> src/main.rs:5:13
  |
5 |     let _ = x as *const u32;
  |             ^^^^^^^^^^^^^^^
  |
note: lint level defined here
 --> src/main.rs:1:9
  |
1 | #![deny(trivial_casts)]
  |         ^^^^^^^^^^^^^
```

## trivial-numeric-casts

This lint detects trivial casts of numeric types which could be removed. Some
example code that triggers this lint:

```rust
let x = 42i32 as i32;
```

When set to 'deny', this will produce:

```text
error: trivial numeric cast: `i32` as `i32`. Cast can be replaced by coercion, this might require type ascription or a temporary variable
 --> src/main.rs:4:13
  |
4 |     let x = 42i32 as i32;
  |             ^^^^^^^^^^^^
  |
```

## unreachable-pub

This lint triggers for `pub` items not reachable from the crate root. Some
example code that triggers this lint:

```rust
mod foo {
    pub mod bar {
        
    }
}
```

When set to 'deny', this will produce:

```text
error: unreachable `pub` item
 --> src/main.rs:4:5
  |
4 |     pub mod bar {
  |     ---^^^^^^^^
  |     |
  |     help: consider restricting its visibility: `pub(crate)`
  |
```

## unsafe-code

This lint catches usage of `unsafe` code. Some example code that triggers this lint:

```rust
fn main() {
    unsafe {

    }
}
```

When set to 'deny', this will produce:

```text
error: usage of an `unsafe` block
 --> src/main.rs:4:5
  |
4 | /     unsafe {
5 | |         
6 | |     }
  | |_____^
  |
```

## unstable-features

This lint is deprecated and no longer used.

## unused-extern-crates

This lint guards against `extern crate` items that are never used. Some
example code that triggers this lint:

```rust,ignore
extern crate semver;
```

When set to 'deny', this will produce:

```text
error: unused extern crate
 --> src/main.rs:3:1
  |
3 | extern crate semver;
  | ^^^^^^^^^^^^^^^^^^^^
  |
```

## unused-import-braces

This lint catches unnecessary braces around an imported item. Some example
code that triggers this lint:

```rust
use test::{A};

pub mod test {
    pub struct A;
}
# fn main() {}
```

When set to 'deny', this will produce:

```text
error: braces around A is unnecessary
 --> src/main.rs:3:1
  |
3 | use test::{A};
  | ^^^^^^^^^^^^^^
  |
```

To fix it, `use test::A;`

## unused-qualifications

This lint detects unnecessarily qualified names. Some example code that triggers this lint:

```rust
mod foo {
    pub fn bar() {}
}

fn main() {
    use foo::bar;
    foo::bar();
}
```

When set to 'deny', this will produce:

```text
error: unnecessary qualification
 --> src/main.rs:9:5
  |
9 |     foo::bar();
  |     ^^^^^^^^
  |
```

You can call `bar()` directly, without the `foo::`.

## unused-results

This lint checks for the unused result of an expression in a statement. Some
example code that triggers this lint:

```rust,no_run
fn foo<T>() -> T { panic!() }

fn main() {
    foo::<usize>();
}
```

When set to 'deny', this will produce:

```text
error: unused result
 --> src/main.rs:6:5
  |
6 |     foo::<usize>();
  |     ^^^^^^^^^^^^^^^
  |
```

## variant-size-differences

This lint detects enums with widely varying variant sizes. Some example code that triggers this lint:

```rust
enum En {
    V0(u8),
    VBig([u8; 1024]),
}
```

When set to 'deny', this will produce:

```text
error: enum variant is more than three times larger (1024 bytes) than the next largest
 --> src/main.rs:5:5
  |
5 |     VBig([u8; 1024]),   //~ ERROR variant is more than three times larger
  |     ^^^^^^^^^^^^^^^^
  |
```
