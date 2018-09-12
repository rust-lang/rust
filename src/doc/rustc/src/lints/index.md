# Lints

In software, a "lint" is a tool used to help improve your source code. The
Rust compiler contains a number of lints, and when it compiles your code, it will
also run the lints. These lints may produce a warning, an error, or nothing at all,
depending on how you've configured things.

Here's a small example:

```bash
$ cat main.rs
fn main() {
    let x = 5;
}
$ rustc main.rs
warning: unused variable: `x`
 --> main.rs:2:9
  |
2 |     let x = 5;
  |         ^
  |
  = note: #[warn(unused_variables)] on by default
  = note: to avoid this warning, consider using `_x` instead
```

This is the `unused_variables` lint, and it tells you that you've introduced
a variable that you don't use in your code. That's not *wrong*, so it's not
an error, but it might be a bug, so you get a warning.
