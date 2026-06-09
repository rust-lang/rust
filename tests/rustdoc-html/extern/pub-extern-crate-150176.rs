//@ aux-build:pub-extern-crate.rs

// A refactor had left us missing the closing tags,
// ensure that they are present.
// https://github.com/rust-lang/rust/issues/150176

//@ has pub_extern_crate_150176/index.html
//@ hasraw - '<dt><code>pub extern crate inner;</code></dt>'
pub extern crate inner;
