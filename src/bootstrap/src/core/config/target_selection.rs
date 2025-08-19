use std::fmt;

use crate::core::config::SplitDebuginfo;
use crate::utils::cache::{INTERNER, Interned};
use crate::{Path, env};

#[derive(Copy, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
// N.B.: This type is used everywhere, and the entire codebase relies on it being Copy.
// Making !Copy is highly nontrivial!
pub struct TargetSelection {
    pub triple: Interned<String>,
    pub file: Option<Interned<String>>,
    pub synthetic: bool,
}

/// Newtype over `Vec<TargetSelection>` so we can implement custom parsing logic
#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct TargetSelectionList(pub Vec<TargetSelection>);

pub fn target_selection_list(s: &str) -> Result<TargetSelectionList, String> {
    Ok(TargetSelectionList(
        s.split(',').filter(|s| !s.is_empty()).map(TargetSelection::from_user).collect(),
    ))
}

impl TargetSelection {
    pub fn from_user(selection: &str) -> Self {
        let path = Path::new(selection);

        let (triple, file) = if path.exists() {
            let triple = path
                .file_stem()
                .expect("Target specification file has no file stem")
                .to_str()
                .expect("Target specification file stem is not UTF-8");

            (triple, Some(selection))
        } else {
            (selection, None)
        };

        let triple = INTERNER.intern_str(triple);
        let file = file.map(|f| INTERNER.intern_str(f));

        Self { triple, file, synthetic: false }
    }

    pub fn create_synthetic(triple: &str, file: &str) -> Self {
        Self {
            triple: INTERNER.intern_str(triple),
            file: Some(INTERNER.intern_str(file)),
            synthetic: true,
        }
    }

    pub fn rustc_target_arg(&self) -> &str {
        self.file.as_ref().unwrap_or(&self.triple)
    }

    pub fn contains(&self, needle: &str) -> bool {
        self.triple.contains(needle)
    }

    pub fn starts_with(&self, needle: &str) -> bool {
        self.triple.starts_with(needle)
    }

    pub fn ends_with(&self, needle: &str) -> bool {
        self.triple.ends_with(needle)
    }

    // See src/bootstrap/synthetic_targets.rs
    pub fn is_synthetic(&self) -> bool {
        self.synthetic
    }

    pub fn is_msvc(&self) -> bool {
        self.contains("msvc")
    }

    pub fn is_windows(&self) -> bool {
        self.contains("windows")
    }

    pub fn is_windows_gnu(&self) -> bool {
        self.ends_with("windows-gnu")
    }

    pub fn is_cygwin(&self) -> bool {
        self.is_windows() &&
        // ref. https://cygwin.com/pipermail/cygwin/2022-February/250802.html
        env::var("OSTYPE").is_ok_and(|v| v.to_lowercase().contains("cygwin"))
    }

    pub fn needs_crt_begin_end(&self) -> bool {
        self.contains("musl") && !self.contains("unikraft")
    }

    /// Path to the file defining the custom target, if any.
    pub fn filepath(&self) -> Option<&Path> {
        self.file.as_ref().map(Path::new)
    }
}

impl fmt::Display for TargetSelection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.triple)?;
        if let Some(file) = self.file {
            write!(f, "({file})")?;
        }
        Ok(())
    }
}

impl fmt::Debug for TargetSelection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl PartialEq<&str> for TargetSelection {
    fn eq(&self, other: &&str) -> bool {
        self.triple == *other
    }
}

// Targets are often used as directory names throughout bootstrap.
// This impl makes it more ergonomics to use them as such.
impl AsRef<Path> for TargetSelection {
    fn as_ref(&self) -> &Path {
        self.triple.as_ref()
    }
}

impl SplitDebuginfo {
    /// Returns the default `-Csplit-debuginfo` value for the current target. See the comment for
    /// `rust.split-debuginfo` in `bootstrap.example.toml`.
    pub fn default_for_platform(target: TargetSelection) -> Self {
        if target.contains("apple") {
            SplitDebuginfo::Unpacked
        } else if target.is_windows() {
            SplitDebuginfo::Packed
        } else {
            SplitDebuginfo::Off
        }
    }
}
