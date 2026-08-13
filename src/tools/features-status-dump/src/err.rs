#[derive(Debug)]
pub enum SemverFlag {
    Before,
    Since,
}

#[derive(Debug)]
pub(crate) enum DumpError {
    NoSources,
    NoVersions,
    StatusConflict,
    BadSemver { cause: String, flag: SemverFlag },
}

impl std::fmt::Display for DumpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg = match self {
            DumpError::NoSources => {
                "No feature sources given. Specify at least one with --library-path or --compiler-path"
            }
            DumpError::NoVersions => "Empty version range: `after..before`.",
            DumpError::StatusConflict => {
                "At most one feature status is allowed to be `required` at once."
            }
            DumpError::BadSemver { cause, flag } => {
                let flag_name = match flag {
                    SemverFlag::Before => "--before",
                    SemverFlag::Since => "--after",
                };
                &format!("Invalid semver {} in flag {}.", cause, flag_name)
            }
        };
        f.pad(msg)
    }
}

impl std::error::Error for DumpError {}
