//! Tidy check to enforce rules about platform-specific code in std.
//!
//! This is intended to maintain existing standards of code
//! organization in hopes that the standard library will continue to
//! be refactored to isolate platform-specific bits, making porting
//! easier; where "standard library" roughly means "all the
//! dependencies of the std and test crates".
//!
//! This generally means placing restrictions on where `cfg(unix)`,
//! `cfg(windows)`, `cfg(target_os)` and `cfg(target_env)` may appear,
//! the basic objective being to isolate platform-specific code to the
//! platform-specific `std::sys` modules, and to the allocation,
//! unwinding, and libc crates.
//!
//! Following are the basic rules, though there are currently
//! exceptions:
//!
//! - core may not have platform-specific code.
//! - libpanic_abort may have platform-specific code.
//! - libpanic_unwind may have platform-specific code.
//! - libunwind may have platform-specific code.
//! - other crates in the std facade may not.
//! - std may have platform-specific code in the following places:
//!   - `sys/unix/`
//!   - `sys/windows/`
//!   - `os/`
//!
//! `std/sys_common` should _not_ contain platform-specific code.
//! Finally, because std contains tests with platform-specific
//! `ignore` attributes, once the parser encounters `mod tests`,
//! platform-specific cfgs are allowed. Not sure yet how to deal with
//! this in the long term.

use std::path::Path;
use std::iter::Iterator;

// Paths that may contain platform-specific code.
const EXCEPTION_PATHS: &[&str] = &[
    // std crates
    "src/libpanic_abort",
    "src/libpanic_unwind",
    "src/libunwind",
    // black_box implementation is LLVM-version specific and it uses
    // target_os to tell targets with different LLVM-versions appart
    // (e.g. `wasm32-unknown-emscripten` vs `wasm32-unknown-unknown`):
    "src/libcore/hint.rs",
    "src/libstd/sys/", // Platform-specific code for std lives here.
                       // This has the trailing slash so that sys_common is not excepted.
    "src/libstd/os", // Platform-specific public interfaces
    "src/rtstartup", // Not sure what to do about this. magic stuff for mingw

    // temporary exceptions
    "src/libstd/lib.rs",
    "src/libstd/path.rs",
    "src/libstd/f32.rs",
    "src/libstd/f64.rs",
    // Integration test for platform-specific run-time feature detection:
    "src/libstd/tests/run-time-detect.rs" ,
    "src/libstd/net/test.rs",
    "src/libstd/sys_common/mod.rs",
    "src/libstd/sys_common/net.rs",
    "src/libstd/sys_common/backtrace.rs",
    "src/libterm", // Not sure how to make this crate portable, but test crate needs it.
    "src/libtest", // Probably should defer to unstable `std::sys` APIs.
    "src/libstd/sync/mpsc", // some tests are only run on non-emscripten

    // std testing crates, okay for now at least
    "src/libcore/tests",
    "src/liballoc/tests/lib.rs",

    // The `VaList` implementation must have platform specific code.
    // The Windows implementation of a `va_list` is always a character
    // pointer regardless of the target architecture. As a result,
    // we must use `#[cfg(windows)]` to conditionally compile the
    // correct `VaList` structure for windows.
    "src/libcore/ffi.rs",

    // non-std crates
    "src/test",
    "src/tools",
    "src/librustc",
    "src/librustdoc",
    "src/libsyntax",
    "src/bootstrap",
];

pub fn check(path: &Path, bad: &mut bool) {
    // Sanity check that the complex parsing here works.
    let mut saw_target_arch = false;
    let mut saw_cfg_bang = false;
    super::walk(path, &mut super::filter_dirs, &mut |entry, contents| {
        let file = entry.path();
        let filestr = file.to_string_lossy().replace("\\", "/");
        if !filestr.ends_with(".rs") { return }

        let is_exception_path = EXCEPTION_PATHS.iter().any(|s| filestr.contains(&**s));
        if is_exception_path { return }

        check_cfgs(contents, &file, bad, &mut saw_target_arch, &mut saw_cfg_bang);
    });

    assert!(saw_target_arch);
    assert!(saw_cfg_bang);
}

fn check_cfgs(contents: &str, file: &Path,
              bad: &mut bool, saw_target_arch: &mut bool, saw_cfg_bang: &mut bool) {
    // For now it's ok to have platform-specific code after 'mod tests'.
    let mod_tests_idx = find_test_mod(contents);
    let contents = &contents[..mod_tests_idx];
    // Pull out all `cfg(...)` and `cfg!(...)` strings.
    let cfgs = parse_cfgs(contents);

    let mut line_numbers: Option<Vec<usize>> = None;
    let mut err = |idx: usize, cfg: &str| {
        if line_numbers.is_none() {
            line_numbers = Some(contents.match_indices('\n').map(|(i, _)| i).collect());
        }
        let line_numbers = line_numbers.as_ref().expect("");
        let line = match line_numbers.binary_search(&idx) {
            Ok(_) => unreachable!(),
            Err(i) => i + 1
        };
        tidy_error!(bad, "{}:{}: platform-specific cfg: {}", file.display(), line, cfg);
    };

    for (idx, cfg) in cfgs {
        // Sanity check that the parsing here works.
        if !*saw_target_arch && cfg.contains("target_arch") { *saw_target_arch = true }
        if !*saw_cfg_bang && cfg.contains("cfg!") { *saw_cfg_bang = true }

        let contains_platform_specific_cfg =
            cfg.contains("target_os")
            || cfg.contains("target_env")
            || cfg.contains("target_vendor")
            || cfg.contains("unix")
            || cfg.contains("windows");

        if !contains_platform_specific_cfg { continue }

        let preceeded_by_doc_comment = {
            let pre_contents = &contents[..idx];
            let pre_newline = pre_contents.rfind('\n');
            let pre_doc_comment = pre_contents.rfind("///");
            match (pre_newline, pre_doc_comment) {
                (Some(n), Some(c)) => n < c,
                (None, Some(_)) => true,
                (_, None) => false,
            }
        };

        if preceeded_by_doc_comment { continue }

        err(idx, cfg);
    }
}

fn find_test_mod(contents: &str) -> usize {
    if let Some(mod_tests_idx) = contents.find("mod tests") {
        // Also capture a previous line indicating that "mod tests" is cfg'd out.
        let prev_newline_idx = contents[..mod_tests_idx].rfind('\n').unwrap_or(mod_tests_idx);
        let prev_newline_idx = contents[..prev_newline_idx].rfind('\n');
        if let Some(nl) = prev_newline_idx {
            let prev_line = &contents[nl + 1 .. mod_tests_idx];
            if prev_line.contains("cfg(all(test, not(target_os")
                || prev_line.contains("cfg(all(test, not(any(target_os") {
                nl
            } else {
                mod_tests_idx
            }
        } else {
            mod_tests_idx
        }
    } else {
        contents.len()
    }
}

fn parse_cfgs<'a>(contents: &'a str) -> Vec<(usize, &'a str)> {
    let candidate_cfgs = contents.match_indices("cfg");
    let candidate_cfg_idxs = candidate_cfgs.map(|(i, _)| i);
    // This is puling out the indexes of all "cfg" strings
    // that appear to be tokens followed by a parenthesis.
    let cfgs = candidate_cfg_idxs.filter(|i| {
        let pre_idx = i.saturating_sub(*i);
        let succeeds_non_ident = !contents.as_bytes().get(pre_idx)
            .cloned()
            .map(char::from)
            .map(char::is_alphanumeric)
            .unwrap_or(false);
        let contents_after = &contents[*i..];
        let first_paren = contents_after.find('(');
        let paren_idx = first_paren.map(|ip| i + ip);
        let preceeds_whitespace_and_paren = paren_idx.map(|ip| {
            let maybe_space = &contents[*i + "cfg".len() .. ip];
            maybe_space.chars().all(|c| char::is_whitespace(c) || c == '!')
        }).unwrap_or(false);

        succeeds_non_ident && preceeds_whitespace_and_paren
    });

    cfgs.flat_map(|i| {
        let mut depth = 0;
        let contents_from = &contents[i..];
        for (j, byte) in contents_from.bytes().enumerate() {
            match byte {
                b'(' => {
                    depth += 1;
                }
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some((i, &contents_from[..=j]));
                    }
                }
                _ => { }
            }
        }

        // if the parentheses are unbalanced just ignore this cfg -- it'll be caught when attempting
        // to run the compiler, and there's no real reason to lint it separately here
        None
    }).collect()
}
