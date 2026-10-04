# `diagnostic_on_unknown`

The tracking issue for this feature is: [#152900](https://github.com/rust-lang/rust/issues/152900)

------------------------

The `diagnostic_on_unknown` feature allows use of the `#[diagnostic::on_unknown]` attribute. It should be
placed on use and module declarations as well as the crate root, though it is not an error to be located in other
positions. This attribute is a hint to the compiler to supplement the error message when the
annotated declaration is involved in a name resolution error.

Format parameters with the given named parameter will be replaced with the following text:

-  `{Unresolved}` — The `SimplePathSegment` of the import path that could not be resolved.
-  `{This}` — The name of the annotated item. On `use` statements this is identical to `{Unresolved}`.

The original error message will not be suppressed but is emitted as a note instead.

### On use declarations

```rust,edition2018,compile_fail,E0432
#![feature(diagnostic_on_unknown)]

#[diagnostic::on_unknown(
    message = "`{Unresolved}` doesn't exist",
    label = "you did something silly here"
)]
use doesnt_exist;
```
This will result in the following error:
```text
error[E0432]: `doesnt_exist` doesn't exist
 --> src/lib.rs:7:5
  |
7 | use doesnt_exist;
  |     ^^^^^^^^^^^^ you did something silly here
  |
  = note: unresolved import `doesnt_exist`

For more information about this error, try `rustc --explain E0432`.
```

### On module declarations

```rust,edition2018,compile_fail,E0432
#![feature(diagnostic_on_unknown)]

#[diagnostic::on_unknown(
    message = "module `{This}` is empty, there is no `{Unresolved}` here",
    label = "can't import something from an empty module"
)]
mod empty {}

use empty::what;
```
This will result in the following error:

```text
error[E0432]: module `empty` is empty, there is no `what` here
 --> src/lib.rs:9:5
  |
9 | use empty::what;
  |     ^^^^^^^----
  |            |
  |            can't import something from an empty module
  |
  = note: unresolved import `empty::what`
```

### On the crate root

This additionally requires the `#![feature(custom_inner_attributes)]` feature:

```rust,edition2018,compile_fail,E0432
#![feature(diagnostic_on_unknown)]
#![feature(custom_inner_attributes)]
#![diagnostic::on_unknown(message = "Say `{Unresolved}` again!")]

use self::what;
```
### Example

Consider the following pair of macros, one which creates a hidden module with a constant and another that reads it.

```rust
#![feature(macro_metavar_expr, macro_attr)]

macro_rules! instrument {
    attr() { $vis:vis fn $fn_name:ident ($($arg_name:ident : $arg_ty:ty),*) $(-> $ret:ty)? $body:block } => {
        $vis fn $fn_name ($($arg_name:$arg_ty),*) $(-> $ret)? $body
        #[doc(hidden)]
        $vis mod $fn_name {
            pub const LEN: usize = ${ count($arg_name) };
        }
    }
}

macro_rules! count_args {
    ($name:ident) => {{
        $name::LEN
    }}
}

#[instrument]
fn add(a: u8, b: u8) -> u8 {
    a + b
}

fn main() {
    let n = count_args!(add);
    println!("`add` has {n} arguments");
}
```

If the `#[instrument]` macro is omitted it will emit this confusing error:
```text
error[E0433]: cannot find module or crate `add` in this scope
  --> src/main.rs:25:25
   |
25 |     let n = count_args!(add);
   |                         ^^^ function `add` is not a crate or module
```

`#[diagnostic::on_unknown]` can be used to customize this error message:

```rust,edition2018,compile_fail,E0432
#![feature(diagnostic_on_unknown)]
# #![feature(macro_metavar_expr, macro_attr)]
#
# #[allow(unused_macros)]
# macro_rules! instrument {
#     attr() { $vis:vis fn $fn_name:ident ($($arg_name:ident : $arg_ty:ty),*) $(-> $ret:ty)? $body:block } => {
#         $vis fn $fn_name ($($arg_name:$arg_ty),*) $(-> $ret)? $body
#         #[doc(hidden)]
#         $vis  mod $fn_name {
#             pub const LEN: usize = ${ count($arg_name) };
#         }
#     }
# }

macro_rules! count_args {
    ($name:ident) => {{
        #[diagnostic::on_unknown(
            message = "cannot count arguments of `{Unresolved}`",
            label = "`{Unresolved}` is not a function decorated \
                    with the `#[instrument]` macro"
        )]
        use $name::LEN as length;
        length
    }}
}

// #[instrument]
fn add(a: u8, b: u8) -> u8 {
  a + b
}

fn main() {
    let n = count_args!(add);
    println!("`add` has {n} arguments");
}
```
This produces:
```text
error[E0432]: cannot count arguments of `add`
  --> src/main.rs:33:25
   |
33 |     let n = count_args!(add);
   |                         ^^^ `add` is not a function decorated with
   |                              the `#[instrument]` macro
   |
   = note: unresolved import `add`
```

### Edition differences

In the 2015 edition, use paths are relative to the crate root. For example, `use empty` will be resolved relative to the crate root and resolution will not encounter the `mod empty {}` declaration.
```rust,edition2015,compile_fail,E0432
#![feature(diagnostic_on_unknown)]

mod foo {
   #[diagnostic::on_unknown(message = "oh oh")]
   mod empty {}

   use empty::what;
}
```

```text
error[E0432]: unresolved import `empty`
 --> src/main.rs:9:8
  |
9 |    use empty::what;
  |        ^^^^^
  |
```
