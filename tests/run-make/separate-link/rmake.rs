// The compiler flags no-link (and by extension, link-only) used to be broken
// due to changes in encoding/decoding. This was patched, and this test checks
// that these flags are not broken again, resulting in successful compilation.
// See https://github.com/rust-lang/rust/issues/77857

//@ ignore-cross-compile

use run_make_support::{run, rustc};

fn main() {
    rustc().stdin_buf(b"fn main(){}").arg("-Zno-link").arg("-").run();
    rustc().arg("-Zlink-only").input("rust_out.rlink").run();
    run("rust_out");
}
