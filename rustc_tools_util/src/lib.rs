#![cfg_attr(feature = "deny-warnings", deny(warnings))]

use std::env;

/// This macro creates the version string during compilation from the
/// current environment
#[macro_export]
macro_rules! get_version_info {
    () => {{
        let major = std::env!("CARGO_PKG_VERSION_MAJOR").parse::<u8>().unwrap();
        let minor = std::env!("CARGO_PKG_VERSION_MINOR").parse::<u8>().unwrap();
        let patch = std::env!("CARGO_PKG_VERSION_PATCH").parse::<u16>().unwrap();
        let crate_name = String::from(std::env!("CARGO_PKG_NAME"));

        let host_compiler = std::option_env!("RUSTC_RELEASE_CHANNEL").map(str::to_string);
        let commit_hash = std::option_env!("GIT_HASH").map(str::to_string);
        let commit_date = std::option_env!("COMMIT_DATE").map(str::to_string);

        $crate::VersionInfo {
            major,
            minor,
            patch,
            host_compiler,
            commit_hash,
            commit_date,
            crate_name,
        }
    }};
}

/// This macro can be used in `build.rs` to automatically set the needed
/// environment values, namely `GIT_HASH`, `COMMIT_DATE` and
/// `RUSTC_RELEASE_CHANNEL`
#[macro_export]
macro_rules! setup_version_info {
    () => {{
        println!(
            "cargo:rustc-env=GIT_HASH={}",
            $crate::get_commit_hash().unwrap_or_default()
        );
        println!(
            "cargo:rustc-env=COMMIT_DATE={}",
            $crate::get_commit_date().unwrap_or_default()
        );
        println!("cargo:rustc-env=RUSTC_RELEASE_CHANNEL={}", $crate::get_channel());
    }};
}

// some code taken and adapted from RLS and cargo
pub struct VersionInfo {
    pub major: u8,
    pub minor: u8,
    pub patch: u16,
    pub host_compiler: Option<String>,
    pub commit_hash: Option<String>,
    pub commit_date: Option<String>,
    pub crate_name: String,
}

impl std::fmt::Display for VersionInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let hash = self.commit_hash.clone().unwrap_or_default();
        let hash_trimmed = hash.trim();

        let date = self.commit_date.clone().unwrap_or_default();
        let date_trimmed = date.trim();

        if (hash_trimmed.len() + date_trimmed.len()) > 0 {
            write!(
                f,
                "{} {}.{}.{} ({hash_trimmed} {date_trimmed})",
                self.crate_name, self.major, self.minor, self.patch,
            )?;
        } else {
            write!(f, "{} {}.{}.{}", self.crate_name, self.major, self.minor, self.patch)?;
        }

        Ok(())
    }
}

impl std::fmt::Debug for VersionInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "VersionInfo {{ crate_name: \"{}\", major: {}, minor: {}, patch: {}",
            self.crate_name, self.major, self.minor, self.patch,
        )?;
        if self.commit_hash.is_some() {
            write!(
                f,
                ", commit_hash: \"{}\", commit_date: \"{}\" }}",
                self.commit_hash.clone().unwrap_or_default().trim(),
                self.commit_date.clone().unwrap_or_default().trim()
            )?;
        } else {
            write!(f, " }}")?;
        }

        Ok(())
    }
}

#[must_use]
pub fn get_commit_hash() -> Option<String> {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|r| String::from_utf8(r.stdout).ok())
}

#[must_use]
pub fn get_commit_date() -> Option<String> {
    std::process::Command::new("git")
        .args(["log", "-1", "--date=short", "--pretty=format:%cd"])
        .output()
        .ok()
        .and_then(|r| String::from_utf8(r.stdout).ok())
}

#[must_use]
pub fn get_channel() -> String {
    match env::var("CFG_RELEASE_CHANNEL") {
        Ok(channel) => channel,
        Err(_) => {
            // if that failed, try to ask rustc -V, do some parsing and find out
            match std::process::Command::new("rustc")
                .arg("-V")
                .output()
                .ok()
                .and_then(|r| String::from_utf8(r.stdout).ok())
            {
                Some(rustc_output) => {
                    if rustc_output.contains("beta") {
                        String::from("beta")
                    } else if rustc_output.contains("stable") {
                        String::from("stable")
                    } else {
                        // default to nightly if we fail to parse
                        String::from("nightly")
                    }
                },
                // default to nightly
                None => String::from("nightly"),
            }
        },
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_struct_local() {
        let vi = get_version_info!();
        assert_eq!(vi.major, 0);
        assert_eq!(vi.minor, 2);
        assert_eq!(vi.patch, 1);
        assert_eq!(vi.crate_name, "rustc_tools_util");
        // hard to make positive tests for these since they will always change
        assert!(vi.commit_hash.is_none());
        assert!(vi.commit_date.is_none());
    }

    #[test]
    fn test_display_local() {
        let vi = get_version_info!();
        assert_eq!(vi.to_string(), "rustc_tools_util 0.2.1");
    }

    #[test]
    fn test_debug_local() {
        let vi = get_version_info!();
        let s = format!("{vi:?}");
        assert_eq!(
            s,
            "VersionInfo { crate_name: \"rustc_tools_util\", major: 0, minor: 2, patch: 1 }"
        );
    }
}
