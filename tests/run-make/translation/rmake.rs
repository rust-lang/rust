//! Smoke test for the rustc diagnostics translation infrastructure.
//!
//! # References
//!
//! - Current tracking issue: <https://github.com/rust-lang/rust/issues/132181>.
//! - Old tracking issue: <https://github.com/rust-lang/rust/issues/100717>
//! - Initial translation infra implementation: <https://github.com/rust-lang/rust/pull/95512>.

// This test uses symbolic links to stub out a fake sysroot to save testing time.
//@ needs-symlink
//@ needs-subprocess

#![deny(warnings)]

use std::path::{Path, PathBuf};

use run_make_support::rustc::sysroot;
use run_make_support::{cwd, rfs, run_in_tmpdir, rustc};

fn main() {
    builtin_fallback_bundle();
    additional_primary_bundle();
    missing_slug_prefers_fallback_bundle();
    broken_primary_bundle_prefers_fallback_bundle();
    locale_sysroot();
    missing_sysroot();
    file_sysroot();
}

/// Check that the test works normally, using the built-in fallback bundle.
fn builtin_fallback_bundle() {
    rustc().input("test.rs").run_fail().assert_stderr_contains("struct literal body without path");
}

/// Check that a primary bundle can be loaded and will be preferentially used where possible.
fn additional_primary_bundle() {
    rustc()
        .input("test.rs")
        .arg("-Ztranslate-additional-ftl=working.ftl")
        .run_fail()
        .assert_stderr_contains("this is a test message");
}

/// Check that a primary bundle without the desired message will use the fallback bundle.
fn missing_slug_prefers_fallback_bundle() {
    rustc()
        .input("test.rs")
        .arg("-Ztranslate-additional-ftl=missing.ftl")
        .run_fail()
        .assert_stderr_contains("struct literal body without path");
}

/// Check that a primary bundle with a broken message (e.g. an interpolated variable is not
/// provided) will use the fallback bundle.
fn broken_primary_bundle_prefers_fallback_bundle() {
    // FIXME(#135817): as of the rmake.rs port, the compiler actually ICEs on the additional
    // `broken.ftl`, even though the original intention seems to be that it should gracefully
    // failover to the fallback bundle. On `aarch64-apple-darwin`, somehow it *doesn't* ICE.

    rustc()
        .env("RUSTC_ICE", "0") // disable ICE dump file, not needed
        .input("test.rs")
        .arg("-Ztranslate-additional-ftl=broken.ftl")
        .run_fail();
}

#[track_caller]
fn shallow_symlink_dir_entries(src_dir: &Path, dst_dir: &Path) {
    for entry in rfs::read_dir(src_dir) {
        let entry = entry.unwrap();
        let src_entry_path = entry.path();
        let src_filename = src_entry_path.file_name().unwrap();
        let meta = rfs::symlink_metadata(&src_entry_path);
        if meta.is_symlink() || meta.is_file() {
            rfs::symlink_file(&src_entry_path, dst_dir.join(src_filename));
        } else if meta.is_dir() {
            rfs::symlink_dir(&src_entry_path, dst_dir.join(src_filename));
        } else {
            unreachable!()
        }
    }
}

#[track_caller]
fn shallow_symlink_dir_entries_materialize_single_dir(
    src_dir: &Path,
    dst_dir: &Path,
    dir_filename: &str,
) {
    shallow_symlink_dir_entries(src_dir, dst_dir);

    let dst_symlink_meta = rfs::symlink_metadata(dst_dir.join(dir_filename));

    if dst_symlink_meta.is_file() || dst_symlink_meta.is_dir() {
        unreachable!();
    }

    #[cfg(windows)]
    {
        use std::os::windows::fs::FileTypeExt as _;
        if dst_symlink_meta.file_type().is_symlink_file() {
            rfs::remove_file(dst_dir.join(dir_filename));
        } else if dst_symlink_meta.file_type().is_symlink_dir() {
            rfs::remove_dir(dst_dir.join(dir_filename));
        } else {
            unreachable!();
        }
    }
    #[cfg(not(windows))]
    {
        rfs::remove_file(dst_dir.join(dir_filename));
    }

    rfs::create_dir_all(dst_dir.join(dir_filename));
}

#[track_caller]
fn setup_fakeroot_parents() -> PathBuf {
    let sysroot = sysroot();
    let fakeroot = cwd().join("fakeroot");
    rfs::create_dir_all(&fakeroot);
    shallow_symlink_dir_entries_materialize_single_dir(&sysroot, &fakeroot, "lib");
    shallow_symlink_dir_entries_materialize_single_dir(
        &sysroot.join("lib"),
        &fakeroot.join("lib"),
        "rustlib",
    );
    shallow_symlink_dir_entries_materialize_single_dir(
        &sysroot.join("lib").join("rustlib"),
        &fakeroot.join("lib").join("rustlib"),
        "src",
    );
    shallow_symlink_dir_entries(
        &sysroot.join("lib").join("rustlib").join("src"),
        &fakeroot.join("lib").join("rustlib").join("src"),
    );
    fakeroot
}

/// Check that a locale can be loaded from the sysroot given a language identifier by making a local
/// copy of the sysroot and adding the custom locale to it.
fn locale_sysroot() {
    run_in_tmpdir(|| {
        let fakeroot = setup_fakeroot_parents();

        // When download-rustc is enabled, real sysroot will have a share directory. Delete the link
        // to it.
        let _ = std::fs::remove_file(fakeroot.join("share"));

        let fake_locale_path = fakeroot.join("share").join("locale").join("zh-CN");
        rfs::create_dir_all(&fake_locale_path);
        rfs::symlink_file(
            cwd().join("working.ftl"),
            fake_locale_path.join("basic-translation.ftl"),
        );

        rustc()
            .env("RUSTC_ICE", "0")
            .input("test.rs")
            .sysroot(&fakeroot)
            .arg("-Ztranslate-lang=zh-CN")
            .run_fail()
            .assert_stderr_contains("this is a test message");
    });
}

/// Check that the compiler errors out when the sysroot requested cannot be found. This test might
/// start failing if there actually exists a Klingon translation of rustc's error messages.
fn missing_sysroot() {
    run_in_tmpdir(|| {
        rustc()
            .input("test.rs")
            .arg("-Ztranslate-lang=tlh")
            .run_fail()
            .assert_stderr_contains("missing locale directory");
    });
}

/// Check that the compiler errors out when the directory for the locale in the sysroot is actually
/// a file.
fn file_sysroot() {
    run_in_tmpdir(|| {
        let fakeroot = setup_fakeroot_parents();
        rfs::create_dir_all(fakeroot.join("share").join("locale"));
        rfs::write(fakeroot.join("share").join("locale").join("zh-CN"), b"not a dir");

        rustc()
            .input("test.rs")
            .sysroot(&fakeroot)
            .arg("-Ztranslate-lang=zh-CN")
            .run_fail()
            .assert_stderr_contains("is not a directory");
    });
}
