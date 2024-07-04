// This test checks the functionality of one of rustdoc's unstable options,
// the ability to specify emit restrictions with `--emit`.
// `invocation-only` should only emit crate-specific files.
// `toolchain-only` should only emit toolchain-specific files.
// `all-shared` should only emit files that can be shared between crates.
// See https://github.com/rust-lang/rust/pull/83478

use run_make_support::{has_extension, has_prefix, rustdoc, shallow_find_files};
use std::path::Path;

fn main() {
    rustdoc()
        .arg("-Zunstable-options")
        .arg("--emit=invocation-specific")
        .output("invocation-only")
        .arg("--resource-suffix=-xxx")
        .args(&["--theme", "y.css"])
        .args(&["--extend-css", "z.css"])
        .input("x.rs")
        .run();
    assert!(Path::new("invocation-only/search-index-xxx.js").exists());
    assert!(Path::new("invocation-only/settings.html").exists());
    assert!(Path::new("invocation-only/x/all.html").exists());
    assert!(Path::new("invocation-only/x/index.html").exists());
    assert!(Path::new("invocation-only/theme-xxx.css").exists()); // generated from z.css
    assert!(!Path::new("invocation-only/storage-xxx.js").exists());
    assert!(!Path::new("invocation-only/SourceSerif4-It.ttf.woff2").exists());
    // FIXME: this probably shouldn't have a suffix
    assert!(Path::new("invocation-only/y-xxx.css").exists());
    // FIXME: this is technically incorrect (see `write_shared`)
    assert!(!Path::new("invocation-only/main-xxx.js").exists());

    rustdoc()
        .arg("-Zunstable-options")
        .arg("--emit=toolchain-shared-resources")
        .output("toolchain-only")
        .arg("--resource-suffix=-xxx")
        .args(&["--extend-css", "z.css"])
        .input("x.rs")
        .run();
    assert_eq!(
        shallow_find_files("toolchain-only/static.files", |path| {
            has_prefix(path, "storage-") && has_extension(path, "js")
        })
        .len(),
        1
    );
    assert_eq!(
        shallow_find_files("toolchain-only/static.files", |path| {
            has_prefix(path, "SourceSerif4-It-") && has_extension(path, "woff2")
        })
        .len(),
        1
    );
    assert_eq!(
        shallow_find_files("toolchain-only/static.files", |path| {
            has_prefix(path, "main-") && has_extension(path, "js")
        })
        .len(),
        1
    );
    assert!(!Path::new("toolchain-only/search-index-xxx.js").exists());
    assert!(!Path::new("toolchain-only/x/index.html").exists());
    assert!(!Path::new("toolchain-only/theme.css").exists());
    assert!(!Path::new("toolchain-only/y-xxx.css").exists());

    rustdoc()
        .arg("-Zunstable-options")
        .arg("--emit=toolchain-shared-resources,unversioned-shared-resources")
        .output("all-shared")
        .arg("--resource-suffix=-xxx")
        .args(&["--extend-css", "z.css"])
        .input("x.rs")
        .run();
    assert_eq!(
        shallow_find_files("all-shared/static.files", |path| {
            has_prefix(path, "storage-") && has_extension(path, "js")
        })
        .len(),
        1
    );
    assert_eq!(
        shallow_find_files("all-shared/static.files", |path| {
            has_prefix(path, "SourceSerif4-It-") && has_extension(path, "woff2")
        })
        .len(),
        1
    );
    assert!(!Path::new("all-shared/search-index-xxx.js").exists());
    assert!(!Path::new("all-shared/settings.html").exists());
    assert!(!Path::new("all-shared/x").exists());
    assert!(!Path::new("all-shared/src").exists());
    assert!(!Path::new("all-shared/theme.css").exists());
    assert_eq!(
        shallow_find_files("all-shared/static.files", |path| {
            has_prefix(path, "main-") && has_extension(path, "js")
        })
        .len(),
        1
    );
    assert!(!Path::new("all-shared/y-xxx.css").exists());
}
