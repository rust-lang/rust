# `diagnostic_on_move`

The tracking issue for this feature is: [#154181]

------------------------

The `diagnostic_on_move` feature allows use of the `#[diagnostic::on_move]` attribute. It should be
placed on struct, enum and union declarations, though it is not an error to be located in other
positions. This attribute is a hint to the compiler to supplement the error message when the
annotated type is involved in a borrowcheck error.

For example, [`File`] is annotated as such:
```rust
#![feature(diagnostic_on_move)]

#[diagnostic::on_move(note = "you can use `File::try_clone` \
                             to duplicate a `File` instance")]
pub struct File {
    // ...
}
```

When you try to use a `File` after it's already been moved, it will helpfully tell you about `try_clone`.

The message and label can also be customized:

```rust
#![feature(diagnostic_on_move)]

use std::marker::PhantomData;

#[diagnostic::on_move(
    message = "`{Self}` cannot be used multiple times",
    label = "this token may only be used once",
    note = "you can create a new `Token` with `Token::conjure()`"
)]
pub struct Token<'brand> {
    spooky: PhantomData<&'brand ()>,
}

impl Token<'_> {
    pub fn conjure<'u>() -> Token<'u> {
        Token {
            spooky: PhantomData,
        }
    }
}
```
The user may try to use it like this:
```rust,compile_fail,E0382
# #![feature(diagnostic_on_move)]
#
# use std::marker::PhantomData;
#
# #[diagnostic::on_move(
#     message = "`{Self}` cannot be used multiple times",
#     label = "this token may only be used once",
#     note = "you can create a new `Token` with `Token::conjure()`"
# )]
# pub struct Token<'brand> {
#     spooky: PhantomData<&'brand ()>,
# }
#
# impl Token<'_> {
#     pub fn conjure<'u>() -> Token<'u> {
#         Token {
#             spooky: PhantomData,
#         }
#     }
# }
# fn main() {
let token = Token::conjure();
let _ = (token, token);
# }
```
This will result in the following error:
```text
error[E0382]: `Token` cannot be used multiple times
  --> src/main.rs:24:21
   |
 1 |     let token = Token::conjure();
   |         ----- this token may only be used once
 2 |     let _ = (token, token);
   |              -----  ^^^^^ value used here after move
   |              |
   |              value moved here
   |
   = note: you can create a new `Token` with `Token::conjure()`
```

[`File`]: https://doc.rust-lang.org/nightly/std/fs/struct.File.html "File in std::fs"
[#154181]: https://github.com/rust-lang/rust/issues/154181 "Tracking Issue for #[diagnostic::on_move]"
