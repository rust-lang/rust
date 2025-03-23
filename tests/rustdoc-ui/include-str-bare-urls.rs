// https://github.com/rust-lang/rust/issues/118549
//
// HEADS UP!
//
// Normally, a line with errors on it will also have a comment
// marking it up as something that needs to generate an error.
//
// The test harness doesn't gather hot comments from the `.md` file.
// Rustdoc will generate an error for the line, and the `.stderr`
// snapshot includes this error, but Compiletest doesn't see it.
//
// If the stderr file changes, make sure the warning points at the URL!

#![deny(rustdoc::bare_urls)]
#![doc=include_str!("auxiliary/include-str-bare-urls.md")]

//~? ERROR this URL is not a hyperlink
