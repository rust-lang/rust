#[derive(Debug)]
pub(crate) enum DumpError {
    NoSources,
    NoVersions,
}

impl std::fmt::Display for DumpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg = match self {
            DumpError::NoSources => {
                "No feature sources given. Specify at least one with --library-path or --compiler-path"
            }
            DumpError::NoVersions => {
                "Empty version range. first_version is older than last_version."
            }
        };
        f.pad(msg)
    }
}

impl std::error::Error for DumpError {}
