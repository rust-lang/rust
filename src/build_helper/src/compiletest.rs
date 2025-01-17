//! Types representing arguments to compiletest.

use std::str::FromStr;
use std::fmt;
pub use self::Mode::*;

macro_rules! string_enum {
    ($(#[$meta:meta])* $vis:vis enum $name:ident { $($variant:ident => $repr:expr,)* }) => {
        $(#[$meta])*
        $vis enum $name {
            $($variant,)*
        }

        impl $name {
            $vis const VARIANTS: &'static [Self] = &[$(Self::$variant,)*];
            $vis const STR_VARIANTS: &'static [&'static str] = &[$(Self::$variant.to_str(),)*];

            $vis const fn to_str(&self) -> &'static str {
                match self {
                    $(Self::$variant => $repr,)*
                }
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::Display::fmt(self.to_str(), f)
            }
        }

        impl FromStr for $name {
            type Err = String;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                match s {
                    $($repr => Ok(Self::$variant),)*
                    _ => Err(format!(concat!("unknown `", stringify!($name), "` variant: `{}`"), s)),
                }
            }
        }
    }
}

// Make the macro visible outside of this module, for tests.
#[cfg(test)]
pub(crate) use string_enum;

string_enum! {
    #[derive(Clone, Copy, PartialEq, Debug)]
    pub enum Mode {
        Pretty => "pretty",
        DebugInfo => "debuginfo",
        Codegen => "codegen",
        Rustdoc => "rustdoc",
        RustdocJson => "rustdoc-json",
        CodegenUnits => "codegen-units",
        Incremental => "incremental",
        RunMake => "run-make",
        Ui => "ui",
        RustdocJs => "rustdoc-js",
        MirOpt => "mir-opt",
        Assembly => "assembly",
        CoverageMap => "coverage-map",
        CoverageRun => "coverage-run",
        Crashes => "crashes",
    }
}

impl Default for Mode {
    fn default() -> Self {
        Mode::Ui
    }
}

impl Mode {
    pub fn aux_dir_disambiguator(self) -> &'static str {
        // Pretty-printing tests could run concurrently, and if they do,
        // they need to keep their output segregated.
        match self {
            Pretty => ".pretty",
            _ => "",
        }
    }

    pub fn output_dir_disambiguator(self) -> &'static str {
        // Coverage tests use the same test files for multiple test modes,
        // so each mode should have a separate output directory.
        match self {
            CoverageMap | CoverageRun => self.to_str(),
            _ => "",
        }
    }
}

string_enum! {
    #[derive(Clone, Copy, PartialEq, Debug, Hash)]
    pub enum PassMode {
        Check => "check",
        Build => "build",
        Run => "run",
    }
}

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub enum FailMode {
    Check,
    Build,
    Run,
}

string_enum! {
    #[derive(Clone, Debug, PartialEq)]
    pub enum CompareMode {
        Polonius => "polonius",
        NextSolver => "next-solver",
        NextSolverCoherence => "next-solver-coherence",
        SplitDwarf => "split-dwarf",
        SplitDwarfSingle => "split-dwarf-single",
    }
}

string_enum! {
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub enum Debugger {
        Cdb => "cdb",
        Gdb => "gdb",
        Lldb => "lldb",
    }
}
