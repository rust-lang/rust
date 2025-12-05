//! A build script dependency to create a Windows resource file for the compiler
//!
//! Uses values from the `CFG_VERSION` and `CFG_RELEASE` environment variables
//! to set the product and file version information in the Windows resource file.
use std::{env, fs, path, process};

/// The template for the Windows resource file.
const RESOURCE_TEMPLATE: &str = include_str!("../rustc.rc.in");

/// A subset of the possible values for the `FILETYPE` field in a Windows resource file
///
/// See the `dwFileType` member of [VS_FIXEDFILEINFO](https://learn.microsoft.com/en-us/windows/win32/api/verrsrc/ns-verrsrc-vs_fixedfileinfo#members)
#[derive(Debug, Clone, Copy)]
#[repr(u32)]
pub enum VersionInfoFileType {
    /// `VFT_APP` - The file is an application.
    App = 0x00000001,
    /// `VFT_DLL` - The file is a dynamic link library.
    Dll = 0x00000002,
}

/// Create and compile a Windows resource file with the product and file version information for the rust compiler.
///
/// Returns the path to the compiled resource file
///
/// Does not emit any cargo directives, the caller is responsible for that.
pub fn compile_windows_resource_file(
    file_stem: &path::Path,
    file_description: &str,
    filetype: VersionInfoFileType,
) -> path::PathBuf {
    let mut resources_dir = path::PathBuf::from(env::var_os("OUT_DIR").unwrap());
    resources_dir.push("resources");
    fs::create_dir_all(&resources_dir).unwrap();

    let resource_compiler = if let Ok(path) = env::var("RUSTC_WINDOWS_RC") {
        path.into()
    } else {
        find_msvc_tools::find_tool(&env::var("CARGO_CFG_TARGET_ARCH").unwrap(), "rc.exe")
            .expect("found rc.exe")
            .path()
            .to_owned()
    };

    let rc_path = resources_dir.join(file_stem.with_extension("rc"));

    write_resource_script_file(&rc_path, file_description, filetype);

    let res_path = resources_dir.join(file_stem.with_extension("res"));

    let status = process::Command::new(resource_compiler)
        .arg("/fo")
        .arg(&res_path)
        .arg(&rc_path)
        .status()
        .expect("can execute resource compiler");
    assert!(status.success(), "rc.exe failed with status {}", status);
    assert!(
        res_path.try_exists().unwrap_or(false),
        "resource file {} was not created",
        res_path.display()
    );
    res_path
}

/// Writes a Windows resource script file for the rust compiler with the product and file version information
/// into `rc_path`
fn write_resource_script_file(
    rc_path: &path::Path,
    file_description: &str,
    filetype: VersionInfoFileType,
) {
    let mut resource_script = RESOURCE_TEMPLATE.to_string();

    // Set the string product and file version to the same thing as `rustc --version`
    let descriptive_version = env::var("CFG_VERSION").unwrap_or("unknown".to_string());

    // Set the product name to "Rust Compiler" or "Rust Compiler (nightly)" etc
    let product_name = product_name(env::var("CFG_RELEASE_CHANNEL").unwrap());

    // For the numeric version we need `major,minor,patch,build`.
    // Extract them from `CFG_RELEASE` which is "major.minor.patch" and a "-dev", "-nightly" or similar suffix
    let cfg_release = env::var("CFG_RELEASE").unwrap();
    // remove the suffix, if present and parse into [`ResourceVersion`]
    let version = parse_version(cfg_release.split("-").next().unwrap_or("0.0.0"))
        .expect("valid CFG_RELEASE version");

    resource_script = resource_script
        .replace("@RUSTC_FILEDESCRIPTION_STR@", file_description)
        .replace("@RUSTC_FILETYPE@", &format!("{}", filetype as u32))
        .replace("@RUSTC_FILEVERSION_QUAD@", &version.to_quad_string())
        .replace("@RUSTC_FILEVERSION_STR@", &descriptive_version)
        .replace("@RUSTC_PRODUCTNAME_STR@", &product_name)
        .replace("@RUSTC_PRODUCTVERSION_QUAD@", &version.to_quad_string())
        .replace("@RUSTC_PRODUCTVERSION_STR@", &descriptive_version);

    fs::write(&rc_path, resource_script)
        .unwrap_or_else(|_| panic!("failed to write resource file {}", rc_path.display()));
}

fn product_name(channel: String) -> String {
    format!(
        "Rust Compiler{}",
        if channel == "stable" { "".to_string() } else { format!(" ({})", channel) }
    )
}

/// Windows resources store versions as four 16-bit integers.
struct ResourceVersion {
    major: u16,
    minor: u16,
    patch: u16,
    build: u16,
}

impl ResourceVersion {
    /// Format the version as a comma-separated string of four integers
    /// as expected by Windows resource scripts for the `FILEVERSION` and `PRODUCTVERSION` fields.
    fn to_quad_string(&self) -> String {
        format!("{},{},{},{}", self.major, self.minor, self.patch, self.build)
    }
}

/// Parse a string in the format "major.minor.patch" into a [`ResourceVersion`].
/// The build is set to 0.
/// Returns `None` if the version string is not in the expected format.
fn parse_version(version: &str) -> Option<ResourceVersion> {
    let mut parts = version.split('.');
    let major = parts.next()?.parse::<u16>().ok()?;
    let minor = parts.next()?.parse::<u16>().ok()?;
    let patch = parts.next()?.parse::<u16>().ok()?;
    if parts.next().is_some() {
        None
    } else {
        Some(ResourceVersion { major, minor, patch, build: 0 })
    }
}
