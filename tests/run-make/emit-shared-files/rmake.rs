// This test checks the functionality of one of rustdoc's unstable options,
// the ability to specify emit restrictions with `--emit`.
// `invocation-only` should only emit crate-specific files.
// `toolchain-only` should only emit toolchain-specific files.
// `all-shared` should only emit files that can be shared between crates.
// See https://github.com/rust-lang/rust/pull/83478

//@ needs-target-std

use run_make_support::{has_extension, has_prefix, path, rustdoc, shallow_find_files};

fn main() {
    rustdoc()
        .arg("-Zunstable-options")
        .arg("--emit=invocation-specific")
        .out_dir("invocation-only")
        .arg("--resource-suffix=-xxx")
        .args(&["--theme", "y.css"])
        .args(&["--extend-css", "z.css"])
        .input("x.rs")
        .run();
    assert!(path("invocation-only/search.index/root-xxx.js").exists());
    assert!(path("invocation-only/crates-xxx.js").exists());
    assert!(path("invocation-only/settings.html").exists());
    assert!(path("invocation-only/x/all.html").exists());
    assert!(path("invocation-only/x/index.html").exists());
    assert!(path("invocation-only/theme-xxx.css").exists()); // generated from z.css
    assert!(!path("invocation-only/storage-xxx.js").exists());
    assert!(!path("invocation-only/SourceSerif4-It.ttf.woff2").exists());
    // FIXME: this probably shouldn't have a suffix
    assert!(path("invocation-only/y-xxx.css").exists());
    // FIXME: this is technically incorrect (see `write_shared`)
    assert!(!path("invocation-only/main-xxx.js").exists());

    rustdoc()
        .arg("-Zunstable-options")
        .arg("--emit=toolchain-shared-resources")
        .out_dir("toolchain-only")
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
    assert!(!path("toolchain-only/search-index-xxx.js").exists());
    assert!(!path("toolchain-only/x/index.html").exists());
    assert!(!path("toolchain-only/theme.css").exists());
    assert!(!path("toolchain-only/y-xxx.css").exists());

    rustdoc()
        .arg("-Zunstable-options")
        .arg("--emit=toolchain-shared-resources,unversioned-shared-resources")
        .out_dir("all-shared")
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
    assert!(!path("all-shared/search-index-xxx.js").exists());
    assert!(!path("all-shared/settings.html").exists());
    assert!(!path("all-shared/x").exists());
    assert!(!path("all-shared/src").exists());
    assert!(!path("all-shared/theme.css").exists());
    assert_eq!(
        shallow_find_files("all-shared/static.files", |path| {
            has_prefix(path, "main-") && has_extension(path, "js")
        })
        .len(),
        1
    );
    assert!(!path("all-shared/y-xxx.css").exists());
}
