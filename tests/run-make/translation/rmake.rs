// Various tests on Fluent bundles, useful to change the compiler's language
// to the one requested by the user. Check each comment header to learn the purpose
// of each test case.
// See https://github.com/rust-lang/rust/pull/95512

//@ needs-symlink

use run_make_support::{path, rfs, rustc};

fn main() {
    builtin_fallback_bundle();
    custom_bundle();
    interpolated_variable_missing();
    desired_message_missing();
    custom_locale_from_sysroot();
    no_locale_in_sysroot();
    locale_in_sysroot_is_invalid();
}

fn builtin_fallback_bundle() {
    // Check that the test works normally, using the built-in fallback bundle.
    rustc().input("test.rs").run_fail().assert_stderr_contains("struct literal body without path");
}

fn custom_bundle() {
    // Check that a primary bundle can be loaded and will be preferentially used
    // where possible.
    rustc()
        .env("RUSTC_TRANSLATION_NO_DEBUG_ASSERT", "1")
        .arg("-Ztranslate-additional-ftl=working.ftl")
        .input("test.rs")
        .run_fail()
        .assert_stderr_contains("this is a test message");
}

fn interpolated_variable_missing() {
    // Check that a primary bundle with a broken message (e.g. a interpolated
    // variable is missing) will use the fallback bundle.
    rustc()
        .env("RUSTC_TRANSLATION_NO_DEBUG_ASSERT", "1")
        .arg("-Ztranslate-additional-ftl=missing.ftl")
        .input("test.rs")
        .run_fail()
        .assert_stderr_contains("struct literal body without path");
}

fn desired_message_missing() {
    // Check that a primary bundle without the desired message will use the fallback
    // bundle.
    rustc()
        .env("RUSTC_TRANSLATION_NO_DEBUG_ASSERT", "1")
        .arg("-Ztranslate-additional-ftl=broken.ftl")
        .input("test.rs")
        .run_fail()
        .assert_stderr_contains("struct literal body without path");
}

fn custom_locale_from_sysroot() {
    // Check that a locale can be loaded from the sysroot given a language
    // identifier by making a local copy of the sysroot and adding the custom locale
    // to it.
    let sysroot =
        rustc().env("RUSTC_TRANSLATION_NO_DEBUG_ASSERT", "1").print("sysroot").run().stdout_utf8();
    let sysroot = sysroot.trim();
    rfs::create_dir("fakeroot");
    symlink_all_entries(&sysroot, "fakeroot");
    rfs::remove_file("fakeroot/lib");
    rfs::create_dir("fakeroot/lib");
    symlink_all_entries(path(&sysroot).join("lib"), "fakeroot/lib");
    rfs::remove_file("fakeroot/lib/rustlib");
    rfs::create_dir("fakeroot/lib/rustlib");
    symlink_all_entries(path(&sysroot).join("lib/rustlib"), "fakeroot/lib/rustlib");
    rfs::remove_file("fakeroot/lib/rustlib/src");
    rfs::create_dir("fakeroot/lib/rustlib/src");
    symlink_all_entries(path(&sysroot).join("lib/rustlib/src"), "fakeroot/lib/rustlib/src");
    // When download-rustc is enabled, `sysroot` will have a share directory. Delete the link to it.
    if path("fakeroot/share").exists() {
        rfs::remove_file("fakeroot/share");
    }
    rfs::create_dir_all("fakeroot/share/locale/zh-CN");
    rfs::create_symlink("working.ftl", "fakeroot/share/locale/zh-CN/basic-translation.ftl");
    rustc()
        .env("RUSTC_TRANSLATION_NO_DEBUG_ASSERT", "1")
        .arg("-Ztranslate-lang=zh-CN")
        .input("test.rs")
        .sysroot("fakeroot")
        .run_fail()
        .assert_stderr_contains("this is a test message");
}

fn no_locale_in_sysroot() {
    // Check that the compiler errors out when the sysroot requested cannot be
    // found. This test might start failing if there actually exists a Klingon
    // translation of rustc's error messages.
    rustc()
        .env("RUSTC_TRANSLATION_NO_DEBUG_ASSERT", "1")
        .arg("-Ztranslate-lang=tlh")
        // .input("test.rs")
        .run_fail()
        .assert_stderr_contains("missing locale directory");
}

fn locale_in_sysroot_is_invalid() {
    // Check that the compiler errors out when the directory for the locale in the
    // sysroot is actually a file.
    rfs::remove_dir_all("fakeroot/share/locale/zh-CN");
    rfs::create_file("fakeroot/share/locale/zh-CN");
    rustc()
        .env("RUSTC_TRANSLATION_NO_DEBUG_ASSERT", "1")
        .arg("-Ztranslate-lang=zh-CN")
        .input("test.rs")
        .sysroot("fakeroot")
        .run_fail()
        .assert_stderr_contains("`$sysroot/share/locales/$locale` is not a directory");
}

fn symlink_all_entries<P: AsRef<std::path::Path>>(dir: P, fakepath: &str) {
    for found_path in rfs::shallow_find_dir_entries(dir) {
        rfs::create_symlink(&found_path, path(fakepath).join(found_path.file_name().unwrap()));
    }
}
