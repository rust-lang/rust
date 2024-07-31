// This test checks if internal compilation error (ICE) log files work as expected.
// - Get the number of lines from the log files without any configuration options,
// then check that the line count doesn't change if the backtrace gets configured to be short
// or full.
// - Check that disabling ICE logging results in zero files created.
// - Check that the ICE files contain some of the expected strings.
// - exercise the -Zmetrics-dir nightly flag
// - verify what happens when both the nightly flag and env variable are set
// - test the RUST_BACKTRACE=0 behavior against the file creation

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
    assert_eq!(default, default_set);
    assert!(default > 0);
    // Some of the expected strings in an ICE file should appear.
    assert!(content.contains("thread 'rustc' panicked at"));
    assert!(content.contains("stack backtrace:"));

    test_backtrace_short(default);
    test_backtrace_full(default);
    test_backtrace_disabled(default);

    clear_ice_files();
    // The ICE dump is explicitly disabled. Therefore, this should produce no files.
    rustc().env("RUSTC_ICE", "0").input("lib.rs").arg("-Ztreat-err-as-bug=1").run_fail();
    let ice_files = shallow_find_files(cwd(), |path| {
        has_prefix(path, "rustc-ice") && has_extension(path, "txt")
    });
    assert!(ice_files.is_empty()); // There should be 0 ICE files.

    metrics_dir(default);
}

fn test_backtrace_short(baseline: usize) {
    clear_ice_files();
    rustc()
        .env("RUSTC_ICE", cwd())
        .input("lib.rs")
        .env("RUST_BACKTRACE", "short")
        .arg("-Ztreat-err-as-bug=1")
        .run_fail();
    let short = get_text_from_ice(cwd()).lines().count();
    // backtrace length in dump shouldn't be changed by RUST_BACKTRACE
    assert_eq!(short, baseline);
}

fn test_backtrace_full(baseline: usize) {
    clear_ice_files();
    rustc()
        .env("RUSTC_ICE", cwd())
        .input("lib.rs")
        .env("RUST_BACKTRACE", "full")
        .arg("-Ztreat-err-as-bug=1")
        .run_fail();
    let full = get_text_from_ice(cwd()).lines().count();
    // backtrace length in dump shouldn't be changed by RUST_BACKTRACE
    assert_eq!(full, baseline);
}

fn test_backtrace_disabled(baseline: usize) {
    clear_ice_files();
    rustc()
        .env("RUSTC_ICE", cwd())
        .input("lib.rs")
        .env("RUST_BACKTRACE", "0")
        .arg("-Ztreat-err-as-bug=1")
        .run_fail();
    let disabled = get_text_from_ice(cwd()).lines().count();
    // backtrace length in dump shouldn't be changed by RUST_BACKTRACE
    assert_eq!(disabled, baseline);
}

fn metrics_dir(baseline: usize) {
    test_flag_only(baseline);
    test_flag_and_env(baseline);
}

fn test_flag_only(baseline: usize) {
    clear_ice_files();
    let metrics_arg = format!("-Zmetrics-dir={}", cwd().display());
    rustc().input("lib.rs").arg("-Ztreat-err-as-bug=1").arg(metrics_arg).run_fail();
    let output = get_text_from_ice(cwd()).lines().count();
    assert_eq!(output, baseline);
}

fn test_flag_and_env(baseline: usize) {
    clear_ice_files();
    let metrics_arg = format!("-Zmetrics-dir={}", cwd().display());
    let real_dir = cwd().join("actually_put_ice_here");
    rfs::create_dir(real_dir.clone());
    rustc()
        .input("lib.rs")
        .env("RUSTC_ICE", real_dir.clone())
        .arg("-Ztreat-err-as-bug=1")
        .arg(metrics_arg)
        .run_fail();
    let output = get_text_from_ice(real_dir).lines().count();
    assert_eq!(output, baseline);
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
