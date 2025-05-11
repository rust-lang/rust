use regex::Regex;

use crate::{Rustc, is_msvc};

/// Asserts that `rustc` uses LLD for linking when executed.
pub fn assert_rustc_uses_lld(rustc: &mut Rustc) {
    let stderr = get_stderr_with_linker_messages(rustc);
    assert!(
        has_lld_version_in_logs(&stderr),
        "LLD version should be present in rustc stderr:\n{stderr}"
    );
}

/// Asserts that `rustc` doesn't use LLD for linking when executed.
pub fn assert_rustc_doesnt_use_lld(rustc: &mut Rustc) {
    let stderr = get_stderr_with_linker_messages(rustc);
    assert!(
        !has_lld_version_in_logs(&stderr),
        "LLD version should NOT be present in rustc stderr:\n{stderr}"
    );
}

fn get_stderr_with_linker_messages(rustc: &mut Rustc) -> String {
    // lld-link is used if msvc, otherwise a gnu-compatible lld is used.
    let linker_version_flag = if is_msvc() { "--version" } else { "-Wl,-v" };

    let output = rustc.arg("-Wlinker-messages").link_arg(linker_version_flag).run();
    output.stderr_utf8()
}

fn has_lld_version_in_logs(stderr: &str) -> bool {
    // Strip the `-Wlinker-messages` wrappers prefixing the linker output.
    let stderr = Regex::new(r"warning: linker std(out|err):").unwrap().replace_all(&stderr, "");
    let lld_version_re = Regex::new(r"^LLD [0-9]+\.[0-9]+\.[0-9]+").unwrap();
    stderr.lines().any(|line| lld_version_re.is_match(line.trim()))
}
