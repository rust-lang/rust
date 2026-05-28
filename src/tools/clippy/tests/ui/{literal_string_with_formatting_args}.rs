//@check-pass

// Regression test for <https://github.com/rust-lang/rust-clippy/issues/13885>.
// The `dbg` macro generates a literal with the name of the current file, so
// we need to ensure the lint is not emitted in this case.

// Clippy sets `-Zflatten_format_args=no`, which changes the default behavior of how format args
// are lowered and only that one has this non-macro span. Adding the flag makes it repro on
// godbolt and shows a root context span for the file name string.
//
// So instead of having:
//
// ```
// Lit(
//     Spanned {
//         node: Str(
//             "[/app/example.rs:2:5] \"something\" = ",
//             Cooked,
//         ),
//         span: /rustc/eb54a50837ad4bcc9842924f27e7287ca66e294c/library/std/src/macros.rs:365:35: 365:58 (#4),
//     },
// ),
// ```
//
// We get:
//
// ```
// Lit(
//     Spanned {
//         node: Str(
//             "/app/example.rs",
//             Cooked,
//         ),
//         span: /app/example.rs:2:5: 2:22 (#0),
//     },
// )
// ```

#![crate_name = "foo"]
#![allow(unused)]
#![warn(clippy::literal_string_with_formatting_args)]

fn another_bad() {
    let literal_string_with_formatting_args = 0;
    dbg!("something");
}

fn main() {}
