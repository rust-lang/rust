//! This test checks rustdoc `-` (stdin) handling

use run_make_support::rustdoc;
use std::path::PathBuf;

static INPUT: &str = r#"
//! ```
//! dbg!(());
//! ```
pub struct F;
"#;

fn main() {
    let out_dir = PathBuf::from("doc");

    // rustdoc -
    rustdoc().arg("-").out_dir(&out_dir).stdin(INPUT).run();
    assert!(out_dir.join("rust_out/struct.F.html").try_exists().unwrap());

    // rustdoc --test -
    rustdoc().arg("--test").arg("-").stdin(INPUT).run();

    // rustdoc file.rs -
    rustdoc().arg("file.rs").arg("-").run_fail();
}
