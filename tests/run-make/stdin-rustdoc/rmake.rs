//! This test checks rustdoc `-` (stdin) handling

use run_make_support::{rustdoc, tmp_dir};

static INPUT: &str = r#"
//! ```
//! dbg!(());
//! ```
pub struct F;
"#;

fn main() {
    let tmp_dir = tmp_dir();
    let out_dir = tmp_dir.join("doc");

    // rustdoc -
    rustdoc().arg("-").out_dir(&out_dir).stdin(INPUT).run();
    assert!(out_dir.join("rust_out/struct.F.html").try_exists().unwrap());

    // rustdoc --test -
    rustdoc().arg("--test").arg("-").stdin(INPUT).run();

    // rustdoc file.rs -
    rustdoc().arg("file.rs").arg("-").run_fail();
}
