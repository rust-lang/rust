//! Code for representing rust-analyzer's release version number.

use std::fmt;

/// Information about the git repository where rust-analyzer was built from.
pub struct CommitInfo {
    pub short_commit_hash: &'static str,
    pub commit_hash: &'static str,
    pub commit_date: &'static str,
}

/// Cargo's version.
pub struct VersionInfo {
    /// rust-analyzer's version, such as "1.57.0", "1.58.0-beta.1", "1.59.0-nightly", etc.
    pub version: &'static str,
    /// The release channel we were built for (stable/beta/nightly/dev).
    ///
    /// `None` if not built via bootstrap.
    pub release_channel: Option<&'static str>,
    /// Information about the Git repository we may have been built from.
    ///
    /// `None` if not built from a git repo.
    pub commit_info: Option<CommitInfo>,
}

impl fmt::Display for VersionInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.version)?;

        if let Some(ci) = &self.commit_info {
            write!(f, " ({} {})", ci.short_commit_hash, ci.commit_date)?;
        };
        Ok(())
    }
}

/// Returns information about cargo's version.
pub const fn version() -> VersionInfo {
    let version = match option_env!("CFG_RELEASE") {
        Some(x) => x,
        None => "0.0.0",
    };

    let release_channel = option_env!("CFG_RELEASE_CHANNEL");
    let commit_info = match (
        option_env!("RA_COMMIT_SHORT_HASH"),
        option_env!("RA_COMMIT_HASH"),
        option_env!("RA_COMMIT_DATE"),
    ) {
        (Some(short_commit_hash), Some(commit_hash), Some(commit_date)) => {
            Some(CommitInfo { short_commit_hash, commit_hash, commit_date })
        }
        _ => None,
    };

    VersionInfo { version, release_channel, commit_info }
}
