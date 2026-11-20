use std::fmt;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::str::FromStr;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Profile {
    Compiler,
    Library,
    Tools,
    Dist,
    None,
}

pub(super) static PROFILE_DIR: &str = "src/bootstrap/defaults";

impl Profile {
    pub(super) fn include_path(&self, src_path: &Path) -> PathBuf {
        PathBuf::from(format!("{}/{PROFILE_DIR}/bootstrap.{}.toml", src_path.display(), self))
    }

    pub fn all() -> impl Iterator<Item = Self> {
        use Profile::*;
        // N.B. these are ordered by how they are displayed, not alphabetically
        [Library, Compiler, Tools, Dist, None].iter().copied()
    }

    pub fn purpose(&self) -> String {
        use Profile::*;
        match self {
            Library => "Contribute to the standard library",
            Compiler => "Contribute to the compiler itself",
            Tools => "Contribute to tools which depend on the compiler, but do not modify it directly (e.g. rustdoc, clippy, miri)",
            Dist => "Install Rust from source",
            None => "Do not modify `bootstrap.toml`"
        }
        .to_string()
    }

    pub fn all_for_help(indent: &str) -> String {
        let mut out = String::new();
        for choice in Profile::all() {
            writeln!(&mut out, "{}{}: {}", indent, choice, choice.purpose()).unwrap();
        }
        out
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Profile::Compiler => "compiler",
            Profile::Library => "library",
            Profile::Tools => "tools",
            Profile::Dist => "dist",
            Profile::None => "none",
        }
    }
}

impl FromStr for Profile {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "lib" | "library" => Ok(Profile::Library),
            "compiler" => Ok(Profile::Compiler),
            "maintainer" | "dist" | "user" => Ok(Profile::Dist),
            "tools" | "tool" | "rustdoc" | "clippy" | "miri" | "rustfmt" => Ok(Profile::Tools),
            "none" => Ok(Profile::None),
            "llvm" | "codegen" => Err("the \"llvm\" and \"codegen\" profiles have been removed,\
                use \"compiler\" instead which has the same functionality"
                .to_string()),
            _ => Err(format!("unknown profile: '{s}'")),
        }
    }
}

impl fmt::Display for Profile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
