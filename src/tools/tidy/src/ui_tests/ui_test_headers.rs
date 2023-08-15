use core::cmp::Ordering;
use std::fs::File;
use std::io::{self, BufReader};
use std::path::Path;
use test_common::directives::*;
use test_common::{CommentKind, TestComment};

const KNOWN_DIRECTIVES: &[&dyn TestDirective] = [
    &ErrorPatternDirective as _,
    &CompileFlagsDirective as _,
    &AuxBuildDirective as _,
    &RustcEnvDirective as _,
    &RevisionsDirective as _,
    &EditionDirective as _,
    &RunRustfixDirective as _,
    &StderrPerBitwidthDirective as _,
    &CheckPassDirective as _,
    &RunPassDirective as _,
    // FIXME (ui_test): needs-asm-support
]
.as_slice();

/// Check that a file uses ui_test headers if a ui_test version of a header exists.
pub(super) fn check_file_headers(file_path: &Path) -> Result<(), HeaderError> {
    let f = File::open(file_path)?;
    let rdr = BufReader::new(f);

    let mut errors = vec![];

    test_common::iter_header(file_path, rdr, &mut |comment| {
        let line_num = comment.line_num();
        for &directive in KNOWN_DIRECTIVES {
            let directive_match = match_comment(comment, directive);
            // Only one directive will ever match a line, so any path that matches should break
            match directive_match {
                DirectiveMatchResult::NoMatch => {}
                DirectiveMatchResult::NoActionNeeded => {
                    break;
                }
                DirectiveMatchResult::UseUiTestComment => {
                    errors.push(HeaderAction {
                        line_num,
                        line: comment.full_line().to_string(),
                        action: LineAction::UseUiTestComment,
                    });
                    break;
                }
                DirectiveMatchResult::MigrateToUiTest => {
                    errors.push(HeaderAction {
                        line_num,
                        line: comment.full_line().to_string(),
                        action: LineAction::MigrateToUiTest {
                            compiletest_name: directive.compiletest_name().to_string(),
                            ui_test_name: directive.ui_test_name().unwrap().to_string(),
                        },
                    });
                    break;
                }
                DirectiveMatchResult::UseUITestName => {
                    errors.push(HeaderAction {
                        line_num,
                        line: comment.full_line().to_string(),
                        action: LineAction::UseUITestName {
                            compiletest_name: directive.compiletest_name().to_string(),
                            ui_test_name: directive.ui_test_name().unwrap().to_string(),
                        },
                    });
                    break;
                }
            }
        }

        match check_condition(&comment) {
            Ok(_) => {}
            Err(ConditionError::ConvertToUiTest { compiletest_name, ui_test_name }) => {
                errors.push(HeaderAction {
                    line_num,
                    line: comment.full_line().to_string(),
                    action: LineAction::MigrateToUiTest { compiletest_name, ui_test_name },
                })
            }
            Err(ConditionError::UiTestUnknownTarget { target_substr }) => {
                errors.push(HeaderAction {
                    line_num,
                    line: comment.full_line().to_string(),
                    action: LineAction::Error {
                        message: format!("invalid target substring: {}", target_substr),
                    },
                })
            }
            Err(ConditionError::UseUiTestComment) => errors.push(HeaderAction {
                line_num,
                line: comment.full_line().to_string(),
                action: LineAction::UseUiTestComment,
            }),
        }
    });

    if errors.len() > 0 {
        return Err(HeaderError::InvalidHeader { bad_lines: errors });
    }

    Ok(())
}

#[derive(Debug, Clone, Copy)]
enum DirectiveMatchResult {
    /// The directive did not match this comment
    NoMatch,
    /// The directive is known to ui_test and has the correct name. No action
    /// is needed.
    NoActionNeeded,
    /// The directive was a compiletest comment, but it has the right name for
    /// ui_test. It should migrate the comment type without changing the name.
    UseUiTestComment,
    /// The directive was a compiletest comment and should be migrated to a ui_test comment.
    MigrateToUiTest,
    /// The directive was a ui_test style directive, but it was using the compiletest style name.
    /// It must change its name.
    UseUITestName,
}

fn match_comment(comment: TestComment<'_>, directive: &dyn TestDirective) -> DirectiveMatchResult {
    // See the comments on DirectiveMatchResult variants for more information.
    match comment.comment() {
        CommentKind::Compiletest(line) => {
            if line.starts_with(directive.ui_test_name().unwrap())
                && matches!(
                    line.get(
                        directive.ui_test_name().unwrap().len()
                            ..directive.ui_test_name().unwrap().len() + 1
                    ),
                    Some(":" | "\n")
                )
            {
                DirectiveMatchResult::UseUiTestComment
            } else if line.starts_with(directive.compiletest_name())
                && matches!(
                    line.get(
                        directive.compiletest_name().len()..directive.compiletest_name().len() + 1
                    ),
                    Some(":" | "\n")
                )
            {
                DirectiveMatchResult::MigrateToUiTest
            } else {
                DirectiveMatchResult::NoMatch
            }
        }
        CommentKind::UiTest(line) => {
            if line.starts_with(directive.ui_test_name().unwrap())
                && matches!(
                    line.get(
                        directive.ui_test_name().unwrap().len()
                            ..directive.ui_test_name().unwrap().len() + 1
                    ),
                    Some(":" | "\n")
                )
            {
                DirectiveMatchResult::NoActionNeeded
            } else if line.starts_with(directive.compiletest_name())
                && matches!(
                    line.get(
                        directive.compiletest_name().len()..directive.compiletest_name().len() + 1
                    ),
                    Some(":" | "\n")
                )
            {
                DirectiveMatchResult::UseUITestName
            } else {
                DirectiveMatchResult::NoMatch
            }
        }
    }
}

#[derive(Debug, Clone)]
enum ConditionError {
    /// The line should be converted to a ui_test style comment with no other changes.
    UseUiTestComment,
    /// The line should be converted to a ui_test style comment and update its name.
    ConvertToUiTest { compiletest_name: String, ui_test_name: String },
    /// The target substring in the comment is not known.
    UiTestUnknownTarget { target_substr: String },
}

/// Checks that a test comment uses the ui_test style only or ignore directives, and that
/// the value would be known to ui_test at the time of writing (2023-08-13).
fn check_condition(comment: &TestComment<'_>) -> Result<(), ConditionError> {
    let comment_kind = comment.comment();
    // This code only cares about checking that conditions parse, skip non-conditions.
    let Some(condition) = comment_kind
        .parse_name_comment()
        .and_then(|(name, _)| name.strip_prefix("only-").or_else(|| name.strip_prefix("ignore-")))
    else {
        return Ok(());
    };
    match comment.comment() {
        CommentKind::Compiletest(_) => {
            if condition.ends_with("bit") {
                // ui_test accepts the same `Nbit` comments as compiletest.
                Err(ConditionError::UseUiTestComment)
            } else if let Ok(idx) = KNOWN_TARGET_COMPONENTS.binary_search(&condition) {
                // Targets that are known to ui_test should be converted to ui_test style target conditions.
                Err(ConditionError::ConvertToUiTest {
                    compiletest_name: condition.to_owned(),
                    // The index is contained within the array, it was just checked.
                    ui_test_name: format!("target-{}", KNOWN_TARGET_COMPONENTS[idx]),
                })
            } else if condition == "test" {
                Err(ConditionError::UseUiTestComment)
            } else if condition == "wasm32-bare" {
                // wasm32-bare is an alias for wasm32-unknown-unknown
                Err(ConditionError::ConvertToUiTest {
                    compiletest_name: String::from("wasm32-bare"),
                    ui_test_name: String::from("wasm32-unknown-unknown"),
                })
            } else if condition.starts_with("tidy-")
                || ["stable", "beta", "nightly", "stage1", "cross-compile", "remote"]
                    .contains(&condition)
            {
                // Ignore tidy directives, and a few other unknown comment types
                // FIXME (ui_test): make ui_test know about all of these
                Ok(())
            } else {
                // Unknown only/ignore directive, or the target is not known, do nothing.
                eprintln!("unknown comment: {:#?}", comment);
                Ok(())
            }
        }

        CommentKind::UiTest(_) => {
            // See `parse_cfg_name_directive_ui_test` in `src/tools/compiletest/src/header/cfg.rs`,
            // which itself is duplicated from ui_test.
            if let Some(target_substr) =
                condition.strip_prefix("target-").or_else(|| condition.strip_prefix("host-"))
            {
                // Make sure that a target or host exists in the hard coded known hosts.
                // Note: this is *not* something that ui_test does, but it is here to aid in
                // transitioning from compiletest, which does check all known targets.
                if let Ok(_) = KNOWN_TARGET_COMPONENTS.binary_search(&target_substr) {
                    Ok(())
                } else {
                    Err(ConditionError::UiTestUnknownTarget {
                        target_substr: target_substr.to_owned(),
                    })
                }
            } else {
                // The comment was either a known condition that does not need additional
                // validation or the comment was an invalid condition. The test parser will either
                // pass or fail on this comment when it actually tries to handle the tests, so
                // there is no need to do anything here.
                Ok(())
            }
        }
    }
}

#[derive(Debug)]
pub(super) enum LineAction {
    /// The directive was a compiletest comment, but it has the right name for ui_test. It should
    /// migrate the comment type without changing the name.
    UseUiTestComment,
    /// The directive was a compiletest comment and should be migrated to a ui_test comment using
    /// the name specified.
    MigrateToUiTest { compiletest_name: String, ui_test_name: String },
    /// The directive was a ui_test style directive, but it was using the compiletest style name.
    /// It must change its name.
    UseUITestName { compiletest_name: String, ui_test_name: String },
    /// This cannot be automatically fixed, just emit an error.
    Error { message: String },
}

#[derive(Debug)]
pub(super) struct HeaderAction {
    line_num: usize,
    line: String,
    action: LineAction,
}

impl HeaderAction {
    pub const fn line_num(&self) -> usize {
        self.line_num
    }

    pub fn line(&self) -> &str {
        self.line.as_str()
    }

    pub const fn action(&self) -> &LineAction {
        &self.action
    }

    /// A message of the required action, to be used in diagnostics.
    pub fn error_message(&self) -> String {
        match &self.action {
            LineAction::UseUiTestComment => String::from("use a ui_test style //@ comment"),
            LineAction::MigrateToUiTest { ui_test_name, compiletest_name } => {
                format!(
                    "use a ui_test style //@ comment and use the updated name {} instead of {}",
                    ui_test_name, compiletest_name
                )
            }
            LineAction::UseUITestName { compiletest_name, ui_test_name } => {
                format!("use the the updated name {} instead of {}", ui_test_name, compiletest_name)
            }
            LineAction::Error { message } => message.clone(),
        }
    }
}

#[derive(Debug)]
pub(super) enum HeaderError {
    IoError(io::Error),
    InvalidHeader { bad_lines: Vec<HeaderAction> },
}

impl From<io::Error> for HeaderError {
    fn from(value: io::Error) -> Self {
        Self::IoError(value)
    }
}

// All components of target triples (including the whole triple) that were known by compiletest
// at the time of writing (2023-08-13). This list is used to guide migration to ui_test style
// only/ignore directives, since ui_test only knows how to do substring matches on target triples
// where as compiletest does things like target families or environments. In theory this list
// contains at least every directive that's *actually* used (but not all in this list are used).
// This list **must be sorted**, it is used in binary searches.
const KNOWN_TARGET_COMPONENTS: &[&str] = [
    "aarch64",
    "aarch64-apple-darwin",
    "aarch64-apple-ios",
    "aarch64-apple-ios-macabi",
    "aarch64-apple-ios-sim",
    "aarch64-apple-tvos",
    "aarch64-apple-watchos-sim",
    "aarch64-fuchsia",
    "aarch64-kmc-solid_asp3",
    "aarch64-linux-android",
    "aarch64-nintendo-switch-freestanding",
    "aarch64-pc-windows-gnullvm",
    "aarch64-pc-windows-msvc",
    "aarch64-unknown-freebsd",
    "aarch64-unknown-fuchsia",
    "aarch64-unknown-hermit",
    "aarch64-unknown-linux-gnu",
    "aarch64-unknown-linux-gnu_ilp32",
    "aarch64-unknown-linux-musl",
    "aarch64-unknown-linux-ohos",
    "aarch64-unknown-netbsd",
    "aarch64-unknown-none",
    "aarch64-unknown-none-softfloat",
    "aarch64-unknown-nto-qnx710",
    "aarch64-unknown-openbsd",
    "aarch64-unknown-redox",
    "aarch64-unknown-uefi",
    "aarch64-uwp-windows-msvc",
    "aarch64-wrs-vxworks",
    "aarch64_be-unknown-linux-gnu",
    "aarch64_be-unknown-linux-gnu_ilp32",
    "aarch64_be-unknown-netbsd",
    "aix",
    "android",
    "arm",
    "arm-linux-androideabi",
    "arm-unknown-linux-gnueabi",
    "arm-unknown-linux-gnueabihf",
    "arm-unknown-linux-musleabi",
    "arm-unknown-linux-musleabihf",
    "arm64_32-apple-watchos",
    "armeb-unknown-linux-gnueabi",
    "armebv7r-none-eabi",
    "armebv7r-none-eabihf",
    "armv4t-none-eabi",
    "armv4t-unknown-linux-gnueabi",
    "armv5te-none-eabi",
    "armv5te-unknown-linux-gnueabi",
    "armv5te-unknown-linux-musleabi",
    "armv5te-unknown-linux-uclibceabi",
    "armv6-unknown-freebsd",
    "armv6-unknown-netbsd-eabihf",
    "armv6k-nintendo-3ds",
    "armv7-apple-ios",
    "armv7-linux-androideabi",
    "armv7-sony-vita-newlibeabihf",
    "armv7-unknown-freebsd",
    "armv7-unknown-linux-gnueabi",
    "armv7-unknown-linux-gnueabihf",
    "armv7-unknown-linux-musleabi",
    "armv7-unknown-linux-musleabihf",
    "armv7-unknown-linux-ohos",
    "armv7-unknown-linux-uclibceabi",
    "armv7-unknown-linux-uclibceabihf",
    "armv7-unknown-netbsd-eabihf",
    "armv7-wrs-vxworks-eabihf",
    "armv7a-kmc-solid_asp3-eabi",
    "armv7a-kmc-solid_asp3-eabihf",
    "armv7a-none-eabi",
    "armv7a-none-eabihf",
    "armv7k-apple-watchos",
    "armv7r-none-eabi",
    "armv7r-none-eabihf",
    "armv7s-apple-ios",
    "asmjs",
    "asmjs-unknown-emscripten",
    "avr",
    "avr-unknown-gnu-atmega328",
    "bpf",
    "bpfeb-unknown-none",
    "bpfel-unknown-none",
    "cuda",
    "dragonfly",
    "emscripten",
    "espidf",
    "freebsd",
    "fuchsia",
    "gnu",
    "haiku",
    "hermit",
    "hexagon",
    "hexagon-unknown-linux-musl",
    "horizon",
    "i386-apple-ios",
    "i586-pc-nto-qnx700",
    "i586-pc-windows-msvc",
    "i586-unknown-linux-gnu",
    "i586-unknown-linux-musl",
    "i686-apple-darwin",
    "i686-linux-android",
    "i686-pc-windows-gnu",
    "i686-pc-windows-msvc",
    "i686-unknown-freebsd",
    "i686-unknown-haiku",
    "i686-unknown-linux-gnu",
    "i686-unknown-linux-musl",
    "i686-unknown-netbsd",
    "i686-unknown-openbsd",
    "i686-unknown-uefi",
    "i686-uwp-windows-gnu",
    "i686-uwp-windows-msvc",
    "i686-wrs-vxworks",
    "illumos",
    "ios",
    "l4re",
    "linux",
    "loongarch64",
    "loongarch64-unknown-linux-gnu",
    "loongarch64-unknown-none",
    "loongarch64-unknown-none-softfloat",
    "m68k",
    "m68k-unknown-linux-gnu",
    "macos",
    "mips",
    "mips-unknown-linux-gnu",
    "mips-unknown-linux-musl",
    "mips-unknown-linux-uclibc",
    "mips32r6",
    "mips64",
    "mips64-openwrt-linux-musl",
    "mips64-unknown-linux-gnuabi64",
    "mips64-unknown-linux-muslabi64",
    "mips64el-unknown-linux-gnuabi64",
    "mips64el-unknown-linux-muslabi64",
    "mips64r6",
    "mipsel-sony-psp",
    "mipsel-sony-psx",
    "mipsel-unknown-linux-gnu",
    "mipsel-unknown-linux-musl",
    "mipsel-unknown-linux-uclibc",
    "mipsel-unknown-none",
    "mipsisa32r6-unknown-linux-gnu",
    "mipsisa32r6el-unknown-linux-gnu",
    "mipsisa64r6-unknown-linux-gnuabi64",
    "mipsisa64r6el-unknown-linux-gnuabi64",
    "msp430",
    "msp430-none-elf",
    "msvc",
    "musl",
    "netbsd",
    "none",
    "nto",
    "nvptx64",
    "nvptx64-nvidia-cuda",
    "openbsd",
    "powerpc",
    "powerpc-unknown-freebsd",
    "powerpc-unknown-linux-gnu",
    "powerpc-unknown-linux-gnuspe",
    "powerpc-unknown-linux-musl",
    "powerpc-unknown-netbsd",
    "powerpc-unknown-openbsd",
    "powerpc-wrs-vxworks",
    "powerpc-wrs-vxworks-spe",
    "powerpc64",
    "powerpc64-ibm-aix",
    "powerpc64-unknown-freebsd",
    "powerpc64-unknown-linux-gnu",
    "powerpc64-unknown-linux-musl",
    "powerpc64-unknown-openbsd",
    "powerpc64-wrs-vxworks",
    "powerpc64le-unknown-freebsd",
    "powerpc64le-unknown-linux-gnu",
    "powerpc64le-unknown-linux-musl",
    "psp",
    "redox",
    "riscv32",
    "riscv32gc-unknown-linux-gnu",
    "riscv32gc-unknown-linux-musl",
    "riscv32i-unknown-none-elf",
    "riscv32im-unknown-none-elf",
    "riscv32imac-esp-espidf",
    "riscv32imac-unknown-none-elf",
    "riscv32imac-unknown-xous-elf",
    "riscv32imc-esp-espidf",
    "riscv32imc-unknown-none-elf",
    "riscv64",
    "riscv64gc-unknown-freebsd",
    "riscv64gc-unknown-fuchsia",
    "riscv64gc-unknown-hermit",
    "riscv64gc-unknown-linux-gnu",
    "riscv64gc-unknown-linux-musl",
    "riscv64gc-unknown-netbsd",
    "riscv64gc-unknown-none-elf",
    "riscv64gc-unknown-openbsd",
    "riscv64imac-unknown-none-elf",
    "s390x",
    "s390x-unknown-linux-gnu",
    "s390x-unknown-linux-musl",
    "sgx",
    "solaris",
    "solid_asp3",
    "sparc",
    "sparc-unknown-linux-gnu",
    "sparc-unknown-none-elf",
    "sparc64",
    "sparc64-unknown-linux-gnu",
    "sparc64-unknown-netbsd",
    "sparc64-unknown-openbsd",
    "sparcv9-sun-solaris",
    "spirv",
    "thumbv4t-none-eabi",
    "thumbv5te-none-eabi",
    "thumbv6m-none-eabi",
    "thumbv7a-pc-windows-msvc",
    "thumbv7a-uwp-windows-msvc",
    "thumbv7em-none-eabi",
    "thumbv7em-none-eabihf",
    "thumbv7m-none-eabi",
    "thumbv7neon-linux-androideabi",
    "thumbv7neon-unknown-linux-gnueabihf",
    "thumbv7neon-unknown-linux-musleabihf",
    "thumbv8m.base-none-eabi",
    "thumbv8m.main-none-eabi",
    "thumbv8m.main-none-eabihf",
    "tvos",
    "uefi",
    "unknown",
    "vita",
    "vxworks",
    "wasi",
    "wasm",
    "wasm32",
    "wasm32-unknown-emscripten",
    "wasm32-unknown-unknown",
    "wasm32-wasi",
    "wasm64",
    "wasm64-unknown-unknown",
    "watchos",
    "windows",
    "x86",
    "x86_64",
    "x86_64-apple-darwin",
    "x86_64-apple-ios",
    "x86_64-apple-ios-macabi",
    "x86_64-apple-tvos",
    "x86_64-apple-watchos-sim",
    "x86_64-fortanix-unknown-sgx",
    "x86_64-fuchsia",
    "x86_64-linux-android",
    "x86_64-pc-nto-qnx710",
    "x86_64-pc-solaris",
    "x86_64-pc-windows-gnu",
    "x86_64-pc-windows-gnullvm",
    "x86_64-pc-windows-msvc",
    "x86_64-sun-solaris",
    "x86_64-unikraft-linux-musl",
    "x86_64-unknown-dragonfly",
    "x86_64-unknown-freebsd",
    "x86_64-unknown-fuchsia",
    "x86_64-unknown-haiku",
    "x86_64-unknown-hermit",
    "x86_64-unknown-illumos",
    "x86_64-unknown-l4re-uclibc",
    "x86_64-unknown-linux-gnu",
    "x86_64-unknown-linux-gnux32",
    "x86_64-unknown-linux-musl",
    "x86_64-unknown-linux-ohos",
    "x86_64-unknown-netbsd",
    "x86_64-unknown-none",
    "x86_64-unknown-openbsd",
    "x86_64-unknown-redox",
    "x86_64-unknown-uefi",
    "x86_64-uwp-windows-gnu",
    "x86_64-uwp-windows-msvc",
    "x86_64-wrs-vxworks",
    "x86_64h-apple-darwin",
    "xous",
]
.as_slice();

#[allow(dead_code)]
const ENSURE_TARGETS_SORTED: () = {
    if let Err(failed) = static_is_sorted(KNOWN_TARGET_COMPONENTS) {
        const FAIL_MSG: &str = "`KNOWN_TARGET_COMPONENTS` is not sorted at ";
        const_panic(FAIL_MSG, failed);
    }
};

const fn const_slice_cmp(x: &[u8], y: &[u8]) -> Ordering {
    match (x, y) {
        ([], []) => Ordering::Equal,
        // If x is exhausted and y isn't, less.
        ([], [_, ..]) => Ordering::Less,
        // If y is exhausted and x isn't, greater.
        ([_, ..], []) => Ordering::Greater,
        // Compare the head elements of each slice, and if they're equal, compare the tails.
        (&[h1, ref tail1 @ ..], &[h2, ref tail2 @ ..]) => {
            if h1 < h2 {
                Ordering::Less
            } else if h1 > h2 {
                Ordering::Greater
            } else {
                const_slice_cmp(tail1, tail2)
            }
        }
    }
}

const fn static_is_sorted<'a>(slice: &'a [&'a str]) -> Result<(), &'a str> {
    let mut slice = slice;
    while let [a, new_slice @ ..] = slice {
        // Only continue if there's a second element too, otherwise we reached the end successfully.
        let [b, ..] = new_slice else {
            return Ok(());
        };

        // If the first element is greater than the next element, that next element is out of place.
        if matches!(const_slice_cmp(a.as_bytes(), b.as_bytes()), Ordering::Greater) {
            return Err(b);
        }

        // Check the rest of the slice.
        slice = new_slice;
    }

    // Only reachable if there are 0 or 1 elements, which are trivially sorted.
    Ok(())
}

const fn const_panic(msg: &str, append: &str) -> ! {
    // Manual formatting time!
    // Surely no target name is greater than 256-50 ish bytes.
    // Also it would have recursed too deep in `const_slice_cmp`.
    let mut dst_buf = [0_u8; 256];

    let mut idx = 0;
    let msg_bytes = msg.as_bytes();
    while idx < msg_bytes.len() {
        dst_buf[idx] = msg_bytes[idx];
        idx += 1;
    }

    let mut idx = 0;
    let append_bytes = append.as_bytes();
    while idx < append_bytes.len() {
        dst_buf[msg_bytes.len() + idx] = append_bytes[idx];
        idx += 1;
    }

    // SAFETY: The bytes passed in are composed of bytes from &str, so they must be valid UTF8.
    panic!("{}", unsafe { core::str::from_utf8_unchecked(&dst_buf) })
}
