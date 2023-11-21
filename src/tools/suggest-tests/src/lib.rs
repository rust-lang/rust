use std::{
    fmt::{self, Display},
    path::Path,
};

use dynamic_suggestions::DYNAMIC_SUGGESTIONS;
use glob::Pattern;
use static_suggestions::STATIC_SUGGESTIONS;

mod dynamic_suggestions;
mod static_suggestions;

#[cfg(test)]
mod tests;

macro_rules! sug {
    ($cmd:expr) => {
        Suggestion::new($cmd, None, &[])
    };

    ($cmd:expr, $paths:expr) => {
        Suggestion::new($cmd, None, $paths.as_slice())
    };

    ($cmd:expr, $stage:expr, $paths:expr) => {
        Suggestion::new($cmd, Some($stage), $paths.as_slice())
    };
}

pub(crate) use sug;

pub fn get_suggestions<T: AsRef<str>>(modified_files: &[T]) -> Vec<Suggestion> {
    let mut suggestions = Vec::new();

    // static suggestions
    for (globs, sugs) in STATIC_SUGGESTIONS.iter() {
        let globs = globs
            .iter()
            .map(|glob| Pattern::new(glob).expect("Found invalid glob pattern!"))
            .collect::<Vec<_>>();
        let matches_some_glob = |file: &str| globs.iter().any(|glob| glob.matches(file));

        if modified_files.iter().map(AsRef::as_ref).any(matches_some_glob) {
            suggestions.extend_from_slice(sugs);
        }
    }

    // dynamic suggestions
    for sug in DYNAMIC_SUGGESTIONS {
        for file in modified_files {
            let sugs = sug(Path::new(file.as_ref()));

            suggestions.extend_from_slice(&sugs);
        }
    }

    suggestions.sort();
    suggestions.dedup();

    suggestions
}

#[derive(Clone, PartialOrd, Ord, PartialEq, Eq, Debug)]
pub struct Suggestion {
    pub cmd: String,
    pub stage: Option<u32>,
    pub paths: Vec<String>,
}

impl Suggestion {
    pub fn new(cmd: &str, stage: Option<u32>, paths: &[&str]) -> Self {
        Self { cmd: cmd.to_owned(), stage, paths: paths.iter().map(|p| p.to_string()).collect() }
    }

    pub fn with_single_path(cmd: &str, stage: Option<u32>, path: &str) -> Self {
        Self::new(cmd, stage, &[path])
    }
}

impl Display for Suggestion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "{} ", self.cmd)?;

        for path in &self.paths {
            write!(f, "{} ", path)?;
        }

        if let Some(stage) = self.stage {
            write!(f, "{}", stage)?;
        } else {
            // write a sentinel value here (in place of a stage) to be consumed
            // by the shim in bootstrap, it will be read and ignored.
            write!(f, "N/A")?;
        }

        Ok(())
    }
}
