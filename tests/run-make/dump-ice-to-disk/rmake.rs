// This test checks if internal compilation error (ICE) log files work as expected.
// - Get the number of lines from the log files without any configuration options,
// then check that the line count doesn't change if the backtrace gets configured to be short
// or full.
// - Check that disabling ICE logging results in zero files created.
// - Check that the ICE files contain some of the expected strings.
// See https://github.com/rust-lang/rust/pull/108714

use run_make_support::{cwd, has_extension, has_prefix, rfs, rustc, shallow_find_files};

fn main() {
    rustc().input("lib.rs").arg("-Ztreat-err-as-bug=1").run_fail();
    let default = get_text_from_ice(".").lines().count();
    clear_ice_files();

    rustc().env("RUSTC_ICE", cwd()).input("lib.rs").arg("-Ztreat-err-as-bug=1").run_fail();
    let ice_text = get_text_from_ice(cwd());
    let default_set = ice_text.lines().count();
    let content = ice_text;
    let ice_files = shallow_find_files(cwd(), |path| {
        has_prefix(path, "rustc-ice") && has_extension(path, "txt")
    });
    assert_eq!(ice_files.len(), 1); // There should only be 1 ICE file.
    let ice_file_name =
        ice_files.first().and_then(|f| f.file_name()).and_then(|n| n.to_str()).unwrap();
    // Ensure that the ICE dump path doesn't contain `:`, because they cause problems on Windows.
    assert!(!ice_file_name.contains(":"), "{ice_file_name}");

    clear_ice_files();
    rustc()
        .env("RUSTC_ICE", cwd())
        .input("lib.rs")
        .env("RUST_BACKTRACE", "short")
        .arg("-Ztreat-err-as-bug=1")
        .run_fail();
    let short = get_text_from_ice(cwd()).lines().count();
    clear_ice_files();
    rustc()
        .env("RUSTC_ICE", cwd())
        .input("lib.rs")
        .env("RUST_BACKTRACE", "full")
        .arg("-Ztreat-err-as-bug=1")
        .run_fail();
    let full = get_text_from_ice(cwd()).lines().count();
    clear_ice_files();

    // The ICE dump is explicitly disabled. Therefore, this should produce no files.
    rustc().env("RUSTC_ICE", "0").input("lib.rs").arg("-Ztreat-err-as-bug=1").run_fail();
    let ice_files = shallow_find_files(cwd(), |path| {
        has_prefix(path, "rustc-ice") && has_extension(path, "txt")
    });
    assert!(ice_files.is_empty()); // There should be 0 ICE files.

    // The line count should not change.
    assert_eq!(short, default_set);
    assert_eq!(short, default);
    assert_eq!(full, default_set);
    assert!(default > 0);
    // Some of the expected strings in an ICE file should appear.
    assert!(content.contains("thread 'rustc' panicked at"));
    assert!(content.contains("stack backtrace:"));
}

fn clear_ice_files() {
    let ice_files = shallow_find_files(cwd(), |path| {
        has_prefix(path, "rustc-ice") && has_extension(path, "txt")
    });
    for file in ice_files {
        rfs::remove_file(file);
    }
}

#[track_caller]
fn get_text_from_ice(dir: impl AsRef<std::path::Path>) -> String {
    let ice_files =
        shallow_find_files(dir, |path| has_prefix(path, "rustc-ice") && has_extension(path, "txt"));
    assert_eq!(ice_files.len(), 1); // There should only be 1 ICE file.
    let ice_file = ice_files.get(0).unwrap();
    let output = rfs::read_to_string(ice_file);
    output
}
