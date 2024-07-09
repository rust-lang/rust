// This test checks if internal compilation error (ICE) log files work as expected.
// - Get the number of lines from the log files without any configuration options,
// then check that the line count doesn't change if the backtrace gets configured to be short
// or full.
// - Check that disabling ICE logging results in zero files created.
// - Check that the ICE files contain some of the expected strings.
// See https://github.com/rust-lang/rust/pull/108714

// FIXME(Oneirical): try it on Windows!

use run_make_support::{cwd, fs_wrapper, has_extension, has_prefix, rustc, shallow_find_files};

fn main() {
    rustc().input("lib.rs").arg("-Ztreat-err-as-bug=1").run_fail();
    let ice_text = get_text_from_ice();
    let default_set = ice_text.lines().count();
    let content = ice_text;
    // Ensure that the ICE files don't contain `:` in their filename because
    // this causes problems on Windows.
    for file in shallow_find_files(cwd(), |path| {
        has_prefix(path, "rustc-ice") && has_extension(path, "txt")
    }) {
        assert!(!file.display().to_string().contains(":"));
    }

    clear_ice_files();
    rustc().input("lib.rs").env("RUST_BACKTRACE", "short").arg("-Ztreat-err-as-bug=1").run_fail();
    let short = get_text_from_ice().lines().count();
    clear_ice_files();
    rustc().input("lib.rs").env("RUST_BACKTRACE", "full").arg("-Ztreat-err-as-bug=1").run_fail();
    let full = get_text_from_ice().lines().count();
    clear_ice_files();

    // The ICE dump is explicitely disabled. Therefore, this should produce no files.
    rustc().env("RUSTC_ICE", "0").input("lib.rs").arg("-Ztreat-err-as-bug=1").run_fail();
    assert!(get_text_from_ice().is_empty());

    // The line count should not change.
    assert_eq!(short, default_set);
    assert_eq!(full, default_set);
    // Some of the expected strings in an ICE file should appear.
    assert!(content.contains("thread 'rustc' panicked at"));
    assert!(content.contains("stack backtrace:"));
}

fn clear_ice_files() {
    let ice_files = shallow_find_files(cwd(), |path| {
        has_prefix(path, "rustc-ice") && has_extension(path, "txt")
    });
    for file in ice_files {
        fs_wrapper::remove_file(file);
    }
}

fn get_text_from_ice() -> String {
    let ice_files = shallow_find_files(cwd(), |path| {
        has_prefix(path, "rustc-ice") && has_extension(path, "txt")
    });
    let mut output = String::new();
    for file in ice_files {
        output.push_str(&fs_wrapper::read_to_string(file));
    }
    output
}
