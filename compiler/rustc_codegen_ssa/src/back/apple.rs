use std::ffi::OsString;
use std::path::PathBuf;
use std::process::Command;

use itertools::Itertools;
use rustc_middle::middle::exported_symbols::SymbolExportKind;
use rustc_session::Session;
use rustc_target::spec::Target;
pub(super) use rustc_target::spec::apple::OSVersion;
use tracing::debug;

use crate::errors::{XcrunError, XcrunSdkPathWarning};
use crate::fluent_generated as fluent;

#[cfg(test)]
mod tests;

/// The canonical name of the desired SDK for a given target.
pub(super) fn sdk_name(target: &Target) -> &'static str {
    match (&*target.os, &*target.env) {
        ("macos", "") => "MacOSX",
        ("ios", "") => "iPhoneOS",
        ("ios", "sim") => "iPhoneSimulator",
        // Mac Catalyst uses the macOS SDK
        ("ios", "macabi") => "MacOSX",
        ("tvos", "") => "AppleTVOS",
        ("tvos", "sim") => "AppleTVSimulator",
        ("visionos", "") => "XROS",
        ("visionos", "sim") => "XRSimulator",
        ("watchos", "") => "WatchOS",
        ("watchos", "sim") => "WatchSimulator",
        (os, abi) => unreachable!("invalid os '{os}' / abi '{abi}' combination for Apple target"),
    }
}

pub(super) fn macho_platform(target: &Target) -> u32 {
    match (&*target.os, &*target.env) {
        ("macos", _) => object::macho::PLATFORM_MACOS,
        ("ios", "macabi") => object::macho::PLATFORM_MACCATALYST,
        ("ios", "sim") => object::macho::PLATFORM_IOSSIMULATOR,
        ("ios", _) => object::macho::PLATFORM_IOS,
        ("watchos", "sim") => object::macho::PLATFORM_WATCHOSSIMULATOR,
        ("watchos", _) => object::macho::PLATFORM_WATCHOS,
        ("tvos", "sim") => object::macho::PLATFORM_TVOSSIMULATOR,
        ("tvos", _) => object::macho::PLATFORM_TVOS,
        ("visionos", "sim") => object::macho::PLATFORM_XROSSIMULATOR,
        ("visionos", _) => object::macho::PLATFORM_XROS,
        _ => unreachable!("tried to get Mach-O platform for non-Apple target"),
    }
}

/// Add relocation and section data needed for a symbol to be considered
/// undefined by ld64.
///
/// The relocation must be valid, and hence must point to a valid piece of
/// machine code, and hence this is unfortunately very architecture-specific.
///
///
/// # New architectures
///
/// The values here are basically the same as emitted by the following program:
///
/// ```c
/// // clang -c foo.c -target $CLANG_TARGET
/// void foo(void);
///
/// extern int bar;
///
/// void* foobar[2] = {
///     (void*)foo,
///     (void*)&bar,
///     // ...
/// };
/// ```
///
/// Can be inspected with:
/// ```console
/// objdump --macho --reloc foo.o
/// objdump --macho --full-contents foo.o
/// ```
pub(super) fn add_data_and_relocation(
    file: &mut object::write::Object<'_>,
    section: object::write::SectionId,
    symbol: object::write::SymbolId,
    target: &Target,
    kind: SymbolExportKind,
) -> object::write::Result<()> {
    let authenticated_pointer =
        kind == SymbolExportKind::Text && target.llvm_target.starts_with("arm64e");

    let data: &[u8] = match target.pointer_width {
        _ if authenticated_pointer => &[0, 0, 0, 0, 0, 0, 0, 0x80],
        32 => &[0; 4],
        64 => &[0; 8],
        pointer_width => unimplemented!("unsupported Apple pointer width {pointer_width:?}"),
    };

    if target.arch == "x86_64" {
        // Force alignment for the entire section to be 16 on x86_64.
        file.section_mut(section).append_data(&[], 16);
    } else {
        // Elsewhere, the section alignment is the same as the pointer width.
        file.section_mut(section).append_data(&[], target.pointer_width as u64);
    }

    let offset = file.section_mut(section).append_data(data, data.len() as u64);

    let flags = if authenticated_pointer {
        object::write::RelocationFlags::MachO {
            r_type: object::macho::ARM64_RELOC_AUTHENTICATED_POINTER,
            r_pcrel: false,
            r_length: 3,
        }
    } else if target.arch == "arm" {
        // FIXME(madsmtm): Remove once `object` supports 32-bit ARM relocations:
        // https://github.com/gimli-rs/object/pull/757
        object::write::RelocationFlags::MachO {
            r_type: object::macho::ARM_RELOC_VANILLA,
            r_pcrel: false,
            r_length: 2,
        }
    } else {
        object::write::RelocationFlags::Generic {
            kind: object::RelocationKind::Absolute,
            encoding: object::RelocationEncoding::Generic,
            size: target.pointer_width as u8,
        }
    };

    file.add_relocation(section, object::write::Relocation { offset, addend: 0, symbol, flags })?;

    Ok(())
}

pub(super) fn add_version_to_llvm_target(
    llvm_target: &str,
    deployment_target: OSVersion,
) -> String {
    let mut components = llvm_target.split("-");
    let arch = components.next().expect("apple target should have arch");
    let vendor = components.next().expect("apple target should have vendor");
    let os = components.next().expect("apple target should have os");
    let environment = components.next();
    assert_eq!(components.next(), None, "too many LLVM triple components");

    assert!(
        !os.contains(|c: char| c.is_ascii_digit()),
        "LLVM target must not already be versioned"
    );

    let version = deployment_target.fmt_full();
    if let Some(env) = environment {
        // Insert version into OS, before environment
        format!("{arch}-{vendor}-{os}{version}-{env}")
    } else {
        format!("{arch}-{vendor}-{os}{version}")
    }
}

pub(super) fn get_sdk_root(sess: &Session) -> Option<PathBuf> {
    let sdk_name = sdk_name(&sess.target);

    // Attempt to invoke `xcrun` to find the SDK.
    //
    // Note that when cross-compiling from e.g. Linux, the `xcrun` binary may sometimes be provided
    // as a shim by a cross-compilation helper tool. It usually isn't, but we still try nonetheless.
    match xcrun_show_sdk_path(sdk_name, false) {
        Ok((path, stderr)) => {
            // Emit extra stderr, such as if `-verbose` was passed, or if `xcrun` emitted a warning.
            if !stderr.is_empty() {
                sess.dcx().emit_warn(XcrunSdkPathWarning { sdk_name, stderr });
            }
            Some(path)
        }
        Err(err) => {
            // Failure to find the SDK is not a hard error, since the user might have specified it
            // in a manner unknown to us (moreso if cross-compiling):
            // - A compiler driver like `zig cc` which links using an internally bundled SDK.
            // - Extra linker arguments (`-Clink-arg=-syslibroot`).
            // - A custom linker or custom compiler driver.
            //
            // Though we still warn, since such cases are uncommon, and it is very hard to debug if
            // you do not know the details.
            //
            // FIXME(madsmtm): Make this a lint, to allow deny warnings to work.
            // (Or fix <https://github.com/rust-lang/rust/issues/21204>).
            let mut diag = sess.dcx().create_warn(err);
            diag.note(fluent::codegen_ssa_xcrun_about);

            // Recognize common error cases, and give more Rust-specific error messages for those.
            if let Some(developer_dir) = xcode_select_developer_dir() {
                diag.arg("developer_dir", &developer_dir);
                diag.note(fluent::codegen_ssa_xcrun_found_developer_dir);
                if developer_dir.as_os_str().to_string_lossy().contains("CommandLineTools") {
                    if sdk_name != "MacOSX" {
                        diag.help(fluent::codegen_ssa_xcrun_command_line_tools_insufficient);
                    }
                }
            } else {
                diag.help(fluent::codegen_ssa_xcrun_no_developer_dir);
            }

            diag.emit();
            None
        }
    }
}

/// Invoke `xcrun --sdk $sdk_name --show-sdk-path` to get the SDK path.
///
/// The exact logic that `xcrun` uses is unspecified (see `man xcrun` for a few details), and may
/// change between macOS and Xcode versions, but it roughly boils down to finding the active
/// developer directory, and then invoking `xcodebuild -sdk $sdk_name -version` to get the SDK
/// details.
///
/// Finding the developer directory is roughly done by looking at, in order:
/// - The `DEVELOPER_DIR` environment variable.
/// - The `/var/db/xcode_select_link` symlink (set by `xcode-select --switch`).
/// - `/Applications/Xcode.app` (hardcoded fallback path).
/// - `/Library/Developer/CommandLineTools` (hardcoded fallback path).
///
/// Note that `xcrun` caches its result, but with a cold cache this whole operation can be quite
/// slow, especially so the first time it's run after a reboot.
fn xcrun_show_sdk_path(
    sdk_name: &'static str,
    verbose: bool,
) -> Result<(PathBuf, String), XcrunError> {
    // Intentionally invoke the `xcrun` in PATH, since e.g. nixpkgs provide an `xcrun` shim, so we
    // don't want to require `/usr/bin/xcrun`.
    let mut cmd = Command::new("xcrun");
    if verbose {
        cmd.arg("--verbose");
    }
    // The `--sdk` parameter is the same as in xcodebuild, namely either an absolute path to an SDK,
    // or the (lowercase) canonical name of an SDK.
    cmd.arg("--sdk");
    cmd.arg(&sdk_name.to_lowercase());
    cmd.arg("--show-sdk-path");

    // We do not stream stdout/stderr lines directly to the user, since whether they are warnings or
    // errors depends on the status code at the end.
    let output = cmd.output().map_err(|error| XcrunError::FailedInvoking {
        sdk_name,
        command_formatted: format!("{cmd:?}"),
        error,
    })?;

    // It is fine to do lossy conversion here, non-UTF-8 paths are quite rare on macOS nowadays
    // (only possible with the HFS+ file system), and we only use it for error messages.
    let stderr = String::from_utf8_lossy_owned(output.stderr);
    if !stderr.is_empty() {
        debug!(stderr, "original xcrun stderr");
    }

    // Some versions of `xcodebuild` output beefy errors when invoked via `xcrun`,
    // but these are usually red herrings.
    let stderr = stderr
        .lines()
        .filter(|line| {
            !line.contains("Writing error result bundle")
                && !line.contains("Requested but did not find extension point with identifier")
        })
        .join("\n");

    if output.status.success() {
        Ok((stdout_to_path(output.stdout), stderr))
    } else {
        // Output both stdout and stderr, since shims of `xcrun` (such as the one provided by
        // nixpkgs), do not always use stderr for errors.
        let stdout = String::from_utf8_lossy_owned(output.stdout).trim().to_string();
        Err(XcrunError::Unsuccessful {
            sdk_name,
            command_formatted: format!("{cmd:?}"),
            stdout,
            stderr,
        })
    }
}

/// Invoke `xcode-select --print-path`, and return the current developer directory.
///
/// NOTE: We don't do any error handling here, this is only used as a canary in diagnostics (`xcrun`
/// will have already emitted the relevant error information).
fn xcode_select_developer_dir() -> Option<PathBuf> {
    let mut cmd = Command::new("xcode-select");
    cmd.arg("--print-path");
    let output = cmd.output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(stdout_to_path(output.stdout))
}

fn stdout_to_path(mut stdout: Vec<u8>) -> PathBuf {
    // Remove trailing newline.
    if let Some(b'\n') = stdout.last() {
        let _ = stdout.pop().unwrap();
    }
    #[cfg(unix)]
    let path = <OsString as std::os::unix::ffi::OsStringExt>::from_vec(stdout);
    #[cfg(not(unix))] // Not so important, this is mostly used on macOS
    let path = OsString::from(String::from_utf8(stdout).expect("stdout must be UTF-8"));
    PathBuf::from(path)
}
