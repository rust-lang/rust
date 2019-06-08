# Warn-by-default lints

These lints are all set to the 'warn' level by default.

## const-err

This lint detects an erroneous expression while doing constant evaluation. Some
example code that triggers this lint:

```rust,ignore
let b = 200u8 + 200u8;
```

This will produce:

```text
warning: attempt to add with overflow
 --> src/main.rs:2:9
  |
2 | let b = 200u8 + 200u8;
  |         ^^^^^^^^^^^^^
  |
```

## dead-code

This lint detects unused, unexported items. Some
example code that triggers this lint:

```rust
fn foo() {}
```

This will produce:

```text
warning: function is never used: `foo`
 --> src/lib.rs:2:1
  |
2 | fn foo() {}
  | ^^^^^^^^
  |
```

## deprecated

This lint detects use of deprecated items. Some
example code that triggers this lint:

```rust
#[deprecated]
fn foo() {}

fn bar() {
    foo();
}
```

This will produce:

```text
warning: use of deprecated item 'foo'
 --> src/lib.rs:7:5
  |
7 |     foo();
  |     ^^^
  |
```

## illegal-floating-point-literal-pattern

This lint detects floating-point literals used in patterns. Some example code
that triggers this lint:

```rust
let x = 42.0;

match x {
    5.0 => {},
    _ => {},
}
```

This will produce:

```text
warning: floating-point literals cannot be used in patterns
 --> src/main.rs:4:9
  |
4 |         5.0 => {},
  |         ^^^
  |
  = note: #[warn(illegal_floating_point_literal_pattern)] on by default
  = warning: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
  = note: for more information, see issue #41620 <https://github.com/rust-lang/rust/issues/41620>
```

## improper-ctypes

This lint detects proper use of libc types in foreign modules. Some
example code that triggers this lint:

```rust
extern "C" {
    static STATIC: String;
}
```

This will produce:

```text
warning: found struct without foreign-function-safe representation annotation in foreign module, consider adding a #[repr(C)] attribute to the type
 --> src/main.rs:2:20
  |
2 |     static STATIC: String;
  |                    ^^^^^^
  |
```

## late-bound-lifetime-arguments

This lint detects generic lifetime arguments in path segments with
late bound lifetime parameters. Some example code that triggers this lint:

```rust
struct S;

impl S {
    fn late<'a, 'b>(self, _: &'a u8, _: &'b u8) {}
}

fn main() {
    S.late::<'static>(&0, &0);
}
```

This will produce:

```text
warning: cannot specify lifetime arguments explicitly if late bound lifetime parameters are present
 --> src/main.rs:8:14
  |
4 |     fn late<'a, 'b>(self, _: &'a u8, _: &'b u8) {}
  |             -- the late bound lifetime parameter is introduced here
...
8 |     S.late::<'static>(&0, &0);
  |              ^^^^^^^
  |
  = note: #[warn(late_bound_lifetime_arguments)] on by default
  = warning: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
  = note: for more information, see issue #42868 <https://github.com/rust-lang/rust/issues/42868>
```

## non-camel-case-types

This lint detects types, variants, traits and type parameters that don't have
camel case names. Some example code that triggers this lint:

```rust
struct s;
```

This will produce:

```text
warning: type `s` should have a camel case name such as `S`
 --> src/main.rs:1:1
  |
1 | struct s;
  | ^^^^^^^^^
  |
```

## non-shorthand-field-patterns

This lint detects using `Struct { x: x }` instead of `Struct { x }` in a pattern. Some
example code that triggers this lint:

```rust
struct Point {
    x: i32,
    y: i32,
}


fn main() {
    let p = Point {
        x: 5,
        y: 5,
    };

    match p {
        Point { x: x, y: y } => (),
    }
}
```

This will produce:

```text
warning: the `x:` in this pattern is redundant
  --> src/main.rs:14:17
   |
14 |         Point { x: x, y: y } => (),
   |                 --^^
   |                 |
   |                 help: remove this
   |

warning: the `y:` in this pattern is redundant
  --> src/main.rs:14:23
   |
14 |         Point { x: x, y: y } => (),
   |                       --^^
   |                       |
   |                       help: remove this

```

## non-snake-case

This lint detects variables, methods, functions, lifetime parameters and
modules that don't have snake case names. Some example code that triggers
this lint:

```rust
let X = 5;
```

This will produce:

```text
warning: variable `X` should have a snake case name such as `x`
 --> src/main.rs:2:9
  |
2 |     let X = 5;
  |         ^
  |
```

## non-upper-case-globals

This lint detects static constants that don't have uppercase identifiers.
Some example code that triggers this lint:

```rust
static x: i32 = 5;
```

This will produce:

```text
warning: static variable `x` should have an upper case name such as `X`
 --> src/main.rs:1:1
  |
1 | static x: i32 = 5;
  | ^^^^^^^^^^^^^^^^^^
  |
```

## no-mangle-generic-items

This lint detects generic items must be mangled. Some
example code that triggers this lint:

```rust
#[no_mangle]
fn foo<T>(t: T) {

}
```

This will produce:

```text
warning: functions generic over types must be mangled
 --> src/main.rs:2:1
  |
1 |   #[no_mangle]
  |   ------------ help: remove this attribute
2 | / fn foo<T>(t: T) {
3 | |
4 | | }
  | |_^
  |
```

## path-statements

This lint detects path statements with no effect. Some example code that
triggers this lint:

```rust
let x = 42;

x;
```

This will produce:

```text
warning: path statement with no effect
 --> src/main.rs:3:5
  |
3 |     x;
  |     ^^
  |
```

## patterns-in-fns-without-body

This lint detects patterns in functions without body were that were
previously erroneously allowed. Some example code that triggers this lint:

```rust
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
  = note: #[warn(patterns_in_fns_without_body)] on by default
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

## plugin-as-library

This lint detects when compiler plugins are used as ordinary library in
non-plugin crate. Some example code that triggers this lint:

```rust,ignore
#![feature(plugin)]
#![plugin(macro_crate_test)]

extern crate macro_crate_test;
```

## private-in-public

This lint detects private items in public interfaces not caught by the old implementation. Some
example code that triggers this lint:

```rust,ignore
pub trait Trait {
    type A;
}

pub struct S;

mod foo {
    struct Z;

    impl ::Trait for ::S {
        type A = Z;
    }
}
# fn main() {}
```

This will produce:

```text
error[E0446]: private type `foo::Z` in public interface
  --> src/main.rs:11:9
   |
11 |         type A = Z;
   |         ^^^^^^^^^^^ can't leak private type
```

## private-no-mangle-fns

This lint detects functions marked `#[no_mangle]` that are also private.
Given that private functions aren't exposed publicly, and `#[no_mangle]`
controls the public symbol, this combination is erroneous. Some example code
that triggers this lint:

```rust
#[no_mangle]
fn foo() {}
```

This will produce:

```text
warning: function is marked #[no_mangle], but not exported
 --> src/main.rs:2:1
  |
2 | fn foo() {}
  | -^^^^^^^^^^
  | |
  | help: try making it public: `pub`
  |
```

To fix this, either make it public or remove the `#[no_mangle]`.

## private-no-mangle-statics

This lint detects any statics marked `#[no_mangle]` that are private.
Given that private statics aren't exposed publicly, and `#[no_mangle]`
controls the public symbol, this combination is erroneous. Some example code
that triggers this lint:

```rust
#[no_mangle]
static X: i32 = 4;
```

This will produce:

```text
warning: static is marked #[no_mangle], but not exported
 --> src/main.rs:2:1
  |
2 | static X: i32 = 4;
  | -^^^^^^^^^^^^^^^^^
  | |
  | help: try making it public: `pub`
  |
```

To fix this, either make it public or remove the `#[no_mangle]`.

## renamed-and-removed-lints

This lint detects lints that have been renamed or removed. Some
example code that triggers this lint:

```rust
#![deny(raw_pointer_derive)]
```

This will produce:

```text
warning: lint raw_pointer_derive has been removed: using derive with raw pointers is ok
 --> src/main.rs:1:9
  |
1 | #![deny(raw_pointer_derive)]
  |         ^^^^^^^^^^^^^^^^^^
  |
```

To fix this, either remove the lint or use the new name.

## safe-packed-borrows

This lint detects borrowing a field in the interior of a packed structure
with alignment other than 1. Some example code that triggers this lint:

```rust
#[repr(packed)]
pub struct Unaligned<T>(pub T);

pub struct Foo {
    start: u8,
    data: Unaligned<u32>,
}

fn main() {
    let x = Foo { start: 0, data: Unaligned(1) };
    let y = &x.data.0;
}
```

This will produce:

```text
warning: borrow of packed field requires unsafe function or block (error E0133)
  --> src/main.rs:11:13
   |
11 |     let y = &x.data.0;
   |             ^^^^^^^^^
   |
   = note: #[warn(safe_packed_borrows)] on by default
   = warning: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
   = note: for more information, see issue #46043 <https://github.com/rust-lang/rust/issues/46043>
```

## stable-features

This lint detects a `#[feature]` attribute that's since been made stable. Some
example code that triggers this lint:

```rust
#![feature(test_accepted_feature)]
```

This will produce:

```text
warning: this feature has been stable since 1.0.0. Attribute no longer needed
 --> src/main.rs:1:12
  |
1 | #![feature(test_accepted_feature)]
  |            ^^^^^^^^^^^^^^^^^^^^^
  |
```

To fix, simply remove the `#![feature]` attribute, as it's no longer needed.

## type-alias-bounds

This lint detects bounds in type aliases. These are not currently enforced.
Some example code that triggers this lint:

```rust
#[allow(dead_code)]
type SendVec<T: Send> = Vec<T>;
```

This will produce:

```text
warning: bounds on generic parameters are not enforced in type aliases
 --> src/lib.rs:2:17
  |
2 | type SendVec<T: Send> = Vec<T>;
  |                 ^^^^
  |
  = note: #[warn(type_alias_bounds)] on by default
  = help: the bound will not be checked when the type alias is used, and should be removed
```

## tyvar-behind-raw-pointer

This lint detects raw pointer to an inference variable. Some
example code that triggers this lint:

```rust
let data = std::ptr::null();
let _ = &data as *const *const ();

if data.is_null() {}
```

This will produce:

```text
warning: type annotations needed
 --> src/main.rs:4:13
  |
4 |     if data.is_null() {}
  |             ^^^^^^^
  |
  = note: #[warn(tyvar_behind_raw_pointer)] on by default
  = warning: this was previously accepted by the compiler but is being phased out; it will become a hard error in the 2018 edition!
  = note: for more information, see issue #46906 <https://github.com/rust-lang/rust/issues/46906>
```

## unconditional-recursion

This lint detects functions that cannot return without calling themselves.
Some example code that triggers this lint:

```rust
fn foo() {
    foo();
}
```

This will produce:

```text
warning: function cannot return without recursing
 --> src/main.rs:1:1
  |
1 | fn foo() {
  | ^^^^^^^^ cannot return without recursing
2 |     foo();
  |     ----- recursive call site
  |
```

## unions-with-drop-fields

This lint detects use of unions that contain fields with possibly non-trivial drop code. Some
example code that triggers this lint:

```rust
#![feature(untagged_unions)]

union U {
    s: String,
}
```

This will produce:

```text
warning: union contains a field with possibly non-trivial drop code, drop code of union fields is ignored when dropping the union
 --> src/main.rs:4:5
  |
4 |     s: String,
  |     ^^^^^^^^^
  |
```

## unknown-lints

This lint detects unrecognized lint attribute. Some
example code that triggers this lint:

```rust,ignore
#[allow(not_a_real_lint)]
```

This will produce:

```text
warning: unknown lint: `not_a_real_lint`
 --> src/main.rs:1:10
  |
1 | #![allow(not_a_real_lint)]
  |          ^^^^^^^^^^^^^^^
  |
```

## unreachable-code

This lint detects unreachable code paths. Some example code that
triggers this lint:

```rust,no_run
panic!("we never go past here!");

let x = 5;
```

This will produce:

```text
warning: unreachable statement
 --> src/main.rs:4:5
  |
4 |     let x = 5;
  |     ^^^^^^^^^^
  |
```

## unreachable-patterns

This lint detects unreachable patterns. Some
example code that triggers this lint:

```rust
let x = 5;
match x {
    y => (),
    5 => (),
}
```

This will produce:

```text
warning: unreachable pattern
 --> src/main.rs:5:5
  |
5 |     5 => (),
  |     ^
  |
```

The `y` pattern will always match, so the five is impossible to reach.
Remember, match arms match in order, you probably wanted to put the `5` case
above the `y` case.

## unstable-name-collision

This lint detects that you've used a name that the standard library plans to
add in the future, which means that your code may fail to compile without
additional type annotations in the future. Either rename, or add those
annotations now.

## unused-allocation

This lint detects unnecessary allocations that can be eliminated.

## unused-assignments

This lint detects assignments that will never be read. Some
example code that triggers this lint:

```rust
let mut x = 5;
x = 6;
```

This will produce:

```text
warning: value assigned to `x` is never read
 --> src/main.rs:4:5
  |
4 |     x = 6;
  |     ^
  |
```

## unused-attributes

This lint detects attributes that were not used by the compiler. Some
example code that triggers this lint:

```rust
#![macro_export]
```

This will produce:

```text
warning: unused attribute
 --> src/main.rs:1:1
  |
1 | #![macro_export]
  | ^^^^^^^^^^^^^^^^
  |
```

## unused-comparisons

This lint detects comparisons made useless by limits of the types involved. Some
example code that triggers this lint:

```rust
fn foo(x: u8) {
    x >= 0;
}
```

This will produce:

```text
warning: comparison is useless due to type limits
 --> src/main.rs:6:5
  |
6 |     x >= 0;
  |     ^^^^^^
  |
```

## unused-doc-comment

This lint detects doc comments that aren't used by rustdoc. Some
example code that triggers this lint:

```rust
/// docs for x
let x = 12;
```

This will produce:

```text
warning: doc comment not used by rustdoc
 --> src/main.rs:2:5
  |
2 |     /// docs for x
  |     ^^^^^^^^^^^^^^
  |
```

## unused-features

This lint detects unused or unknown features found in crate-level #[feature] directives.
To fix this, simply remove the feature flag.

## unused-imports

This lint detects imports that are never used. Some
example code that triggers this lint:

```rust
use std::collections::HashMap;
```

This will produce:

```text
warning: unused import: `std::collections::HashMap`
 --> src/main.rs:1:5
  |
1 | use std::collections::HashMap;
  |     ^^^^^^^^^^^^^^^^^^^^^^^^^
  |
```

## unused-macros

This lint detects macros that were not used. Some example code that
triggers this lint:

```rust
macro_rules! unused {
    () => {};
}

fn main() {
}
```

This will produce:

```text
warning: unused macro definition
 --> src/main.rs:1:1
  |
1 | / macro_rules! unused {
2 | |     () => {};
3 | | }
  | |_^
  |
```

## unused-must-use

This lint detects unused result of a type flagged as #[must_use]. Some
example code that triggers this lint:

```rust
fn returns_result() -> Result<(), ()> {
    Ok(())
}

fn main() {
    returns_result();
}
```

This will produce:

```text
warning: unused `std::result::Result` that must be used
 --> src/main.rs:6:5
  |
6 |     returns_result();
  |     ^^^^^^^^^^^^^^^^^
  |
```

## unused-mut

This lint detects mut variables which don't need to be mutable. Some
example code that triggers this lint:

```rust
let mut x = 5;
```

This will produce:

```text
warning: variable does not need to be mutable
 --> src/main.rs:2:9
  |
2 |     let mut x = 5;
  |         ----^
  |         |
  |         help: remove this `mut`
  |
```

## unused-parens

This lint detects `if`, `match`, `while` and `return` with parentheses; they
do not need them. Some example code that triggers this lint:

```rust
if(true) {}
```

This will produce:

```text
warning: unnecessary parentheses around `if` condition
 --> src/main.rs:2:7
  |
2 |     if(true) {}
  |       ^^^^^^ help: remove these parentheses
  |
```

## unused-unsafe

This lint detects unnecessary use of an `unsafe` block. Some
example code that triggers this lint:

```rust
unsafe {}
```

This will produce:

```text
warning: unnecessary `unsafe` block
 --> src/main.rs:2:5
  |
2 |     unsafe {}
  |     ^^^^^^ unnecessary `unsafe` block
  |
```

## unused-variables

This lint detects variables which are not used in any way. Some
example code that triggers this lint:

```rust
let x = 5;
```

This will produce:

```text
warning: unused variable: `x`
 --> src/main.rs:2:9
  |
2 |     let x = 5;
  |         ^ help: consider using `_x` instead
  |
```

## warnings

This lint is a bit special; by changing its level, you change every other warning
that would produce a warning to whatever value you'd like:

```rust
#![deny(warnings)]
```

As such, you won't ever trigger this lint in your code directly.

## while-true

This lint detects `while true { }`. Some example code that triggers this
lint:

```rust,no_run
while true {

}
```

This will produce:

```text
warning: denote infinite loops with `loop { ... }`
 --> src/main.rs:2:5
  |
2 |     while true {
  |     ^^^^^^^^^^ help: use `loop`
  |
```
