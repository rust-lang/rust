use std::fmt;

/// Enum of all the "test modes" understood by compiletest.
///
/// Some of these mode names happen to overlap with the names of test suite
/// directories, but the relationship between modes and suites is not 1:1.
/// For example:
/// - Mode `ui` is used by suites `tests/ui` and `tests/rustdoc-ui`
/// - Suite `tests/coverage` uses modes `coverage-map` and `coverage-run`
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum CompiletestMode {
    // tidy-alphabetical-start
    Assembly,
    Codegen,
    CodegenUnits,
    CoverageMap,
    CoverageRun,
    Crashes,
    Debuginfo,
    Incremental,
    MirOpt,
    Pretty,
    RunMake,
    RustdocHtml,
    RustdocJs,
    RustdocJson,
    Ui,
    // tidy-alphabetical-end
}

impl CompiletestMode {
    /// Returns a string representing this mode, which can be passed to
    /// compiletest via a command-line argument.
    ///
    /// These mode names must be kept in sync with the ones understood by
    /// compiletest's `TestMode`, but they change so rarely that doing so
    /// manually should not be burdensome.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            // tidy-alphabetical-start
            Self::Assembly => "assembly",
            Self::Codegen => "codegen",
            Self::CodegenUnits => "codegen-units",
            Self::CoverageMap => "coverage-map",
            Self::CoverageRun => "coverage-run",
            Self::Crashes => "crashes",
            Self::Debuginfo => "debuginfo",
            Self::Incremental => "incremental",
            Self::MirOpt => "mir-opt",
            Self::Pretty => "pretty",
            Self::RunMake => "run-make",
            Self::RustdocHtml => "rustdoc-html",
            Self::RustdocJs => "rustdoc-js",
            Self::RustdocJson => "rustdoc-json",
            Self::Ui => "ui",
            // tidy-alphabetical-end
        }
    }
}

impl fmt::Display for CompiletestMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl fmt::Debug for CompiletestMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <Self as fmt::Display>::fmt(self, f)
    }
}
