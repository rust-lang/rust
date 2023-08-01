use tracing::*;

use crate::{CommentKind, TestComment};

#[derive(Debug, Clone)]
/// A name-value pair from a test directive comment. Variables known by compiletest have **not**
/// been expanded.
pub struct UnexpandedNameValue<'line> {
    name: &'line str,
    value: &'line str,
}

impl<'line> UnexpandedNameValue<'line> {
    fn new(name: &'line str, value: &'line str) -> Self {
        Self { name, value }
    }

    /// Returns the name of the test directive used.
    pub fn name(&self) -> &str {
        self.name
    }

    /// Returns the arguments of the test directive, *without expanding them*.
    /// This should be avoided in favor of `Config::parse_expand_name_value` when possible.
    pub fn unexpanded_value(&self) -> &str {
        self.value
    }
}

pub trait NameValueDirective {
    fn parse_name_value<'line>(
        &self,
        comment: &'line TestComment<'_>,
    ) -> Option<UnexpandedNameValue<'line>>;
}

pub trait NameDirective {
    fn parse_name(&self, comment: &TestComment<'_>) -> bool;
    fn parse_name_negative(&self, comment: &TestComment<'_>) -> bool;
}

pub trait TestDirective {
    fn compiletest_name(&self) -> &'static str;
    fn ui_test_name(&self) -> Option<&'static str>;
}

macro_rules! name_value_directive {
    ($item:ident, $compiletest_name:literal) => {
        #[derive(Debug, Clone, Copy)]
        pub struct $item;

        impl TestDirective for $item {
            fn compiletest_name(&self) -> &'static str {
                $compiletest_name
            }

            fn ui_test_name(&self) -> Option<&'static str> {
                None
            }
        }

        impl NameValueDirective for $item {
            fn parse_name_value<'line>(
                &self,
                comment: &'line TestComment<'_>,
            ) -> Option<UnexpandedNameValue<'line>> {
                match comment.comment() {
                    CommentKind::Compiletest(line) => {
                        if let Some(args) = line
                            .strip_prefix($compiletest_name)
                            .and_then(|with_colon| with_colon.strip_prefix(':'))
                        {
                            debug!("compiletest {}: {}", $compiletest_name, args);
                            Some(UnexpandedNameValue::new($compiletest_name, args))
                        } else {
                            None
                        }
                    }
                    CommentKind::UiTest(_) => None,
                }
            }
        }
    };
    ($item:ident, $compiletest_name:literal, $ui_test_name:literal) => {
        #[derive(Debug, Clone, Copy)]
        pub struct $item;

        impl TestDirective for $item {
            fn compiletest_name(&self) -> &'static str {
                $compiletest_name
            }

            fn ui_test_name(&self) -> Option<&'static str> {
                Some($ui_test_name)
            }
        }

        impl NameValueDirective for $item {
            fn parse_name_value<'line>(
                &self,
                comment: &'line TestComment<'_>,
            ) -> Option<UnexpandedNameValue<'line>> {
                match comment.comment() {
                    CommentKind::Compiletest(line) => {
                        if let Some(args) = line
                            .strip_prefix($compiletest_name)
                            .and_then(|with_colon| with_colon.strip_prefix(':'))
                        {
                            debug!("compiletest {}: {}", $compiletest_name, args);
                            Some(UnexpandedNameValue::new($compiletest_name, args))
                        } else {
                            None
                        }
                    }
                    CommentKind::UiTest(line) => {
                        if let Some(args) = line
                            .strip_prefix($ui_test_name)
                            .and_then(|with_colon| with_colon.strip_prefix(':'))
                        {
                            debug!("ui_test {}: {}", $ui_test_name, args);
                            Some(UnexpandedNameValue::new($ui_test_name, args))
                        } else {
                            None
                        }
                    }
                }
            }
        }
    };
}

macro_rules! name_directive {
    ($item:ident, $compiletest_name:literal) => {
        #[derive(Debug, Clone, Copy)]
        pub struct $item;

        impl TestDirective for $item {
            fn compiletest_name(&self) -> &'static str {
                $compiletest_name
            }

            fn ui_test_name(&self) -> Option<&'static str> {
                None
            }
        }

        impl NameDirective for $item {
            fn parse_name(&self, comment: &TestComment<'_>) -> bool {
                match comment.comment() {
                    CommentKind::Compiletest(line) => {
                        line.strip_prefix($compiletest_name).is_some_and(|rest| {
                            matches!(rest.chars().next(), None | Some(' ') | Some(':'))
                        })
                    }
                    CommentKind::UiTest(_) => false,
                }
            }
            fn parse_name_negative(&self, comment: &TestComment<'_>) -> bool {
                match comment.comment() {
                    CommentKind::Compiletest(line) => {
                        line.strip_prefix("no-").is_some_and(|positive| {
                            positive.strip_prefix($compiletest_name).is_some_and(|rest| {
                                matches!(rest.chars().next(), None | Some(' ') | Some(':'))
                            })
                        })
                    }
                    CommentKind::UiTest(_) => false,
                }
            }
        }
    };
    ($item:ident, $compiletest_name:literal, $ui_test_name:literal) => {
        #[derive(Debug, Clone, Copy)]
        pub struct $item;

        impl TestDirective for $item {
            fn compiletest_name(&self) -> &'static str {
                $compiletest_name
            }

            fn ui_test_name(&self) -> Option<&'static str> {
                Some($ui_test_name)
            }
        }

        impl NameDirective for $item {
            fn parse_name(&self, comment: &TestComment<'_>) -> bool {
                match comment.comment() {
                    CommentKind::Compiletest(line) => {
                        line.strip_prefix($compiletest_name).is_some_and(|rest| {
                            matches!(rest.chars().next(), None | Some(' ') | Some(':'))
                        })
                    }
                    CommentKind::UiTest(line) => {
                        line.strip_prefix($ui_test_name).is_some_and(|rest| {
                            matches!(rest.chars().next(), None | Some(' ') | Some(':'))
                        })
                    }
                }
            }
            fn parse_name_negative(&self, comment: &TestComment<'_>) -> bool {
                match comment.comment() {
                    CommentKind::Compiletest(line) => {
                        line.strip_prefix("no-").is_some_and(|positive| {
                            positive.strip_prefix($compiletest_name).is_some_and(|rest| {
                                matches!(rest.chars().next(), None | Some(' ') | Some(':'))
                            })
                        })
                    }
                    CommentKind::UiTest(line) => line.strip_prefix("no-").is_some_and(|positive| {
                        positive.strip_prefix($ui_test_name).is_some_and(|rest| {
                            matches!(rest.chars().next(), None | Some(' ') | Some(':'))
                        })
                    }),
                }
            }
        }
    };
}

macro_rules! name_val_or_name_directive {
    ($item:ident, $compiletest_name:literal) => {
        #[derive(Debug, Clone, Copy)]
        pub struct $item;

        impl TestDirective for $item {
            fn compiletest_name(&self) -> &'static str {
                $compiletest_name
            }

            fn ui_test_name(&self) -> Option<&'static str> {
                None
            }
        }

        impl NameValueDirective for $item {
            fn parse_name_value<'line>(
                &self,
                comment: &'line TestComment<'_>,
            ) -> Option<UnexpandedNameValue<'line>> {
                match comment.comment() {
                    CommentKind::Compiletest(line) => {
                        if let Some(args) = line
                            .strip_prefix($compiletest_name)
                            .and_then(|with_colon| with_colon.strip_prefix(':'))
                        {
                            debug!("compiletest {}: {}", $compiletest_name, args);
                            Some(UnexpandedNameValue::new($compiletest_name, args))
                        } else {
                            None
                        }
                    }
                    CommentKind::UiTest(_) => None,
                }
            }
        }
        impl NameDirective for $item {
            fn parse_name(&self, comment: &TestComment<'_>) -> bool {
                match comment.comment() {
                    CommentKind::Compiletest(line) => {
                        line.strip_prefix($compiletest_name).is_some_and(|rest| {
                            matches!(rest.chars().next(), None | Some(' ') | Some(':'))
                        })
                    }
                    CommentKind::UiTest(_) => false,
                }
            }
            fn parse_name_negative(&self, comment: &TestComment<'_>) -> bool {
                match comment.comment() {
                    CommentKind::Compiletest(line) => {
                        line.strip_prefix("no-").is_some_and(|positive| {
                            positive.strip_prefix($compiletest_name).is_some_and(|rest| {
                                matches!(rest.chars().next(), None | Some(' ') | Some(':'))
                            })
                        })
                    }
                    CommentKind::UiTest(_) => false,
                }
            }
        }
    };
    ($item:ident, $compiletest_name:literal, $ui_test_name:literal) => {
        #[derive(Debug, Clone, Copy)]
        pub struct $item;

        impl TestDirective for $item {
            fn compiletest_name(&self) -> &'static str {
                $compiletest_name
            }

            fn ui_test_name(&self) -> Option<&'static str> {
                Some($ui_test_name)
            }
        }

        impl NameValueDirective for $item {
            fn parse_name_value<'line>(
                &self,
                comment: &'line TestComment<'_>,
            ) -> Option<UnexpandedNameValue<'line>> {
                match comment.comment() {
                    CommentKind::Compiletest(line) => {
                        if let Some(args) = line
                            .strip_prefix($compiletest_name)
                            .and_then(|with_colon| with_colon.strip_prefix(':'))
                        {
                            debug!("compiletest {}: {}", $compiletest_name, args);
                            Some(UnexpandedNameValue::new($compiletest_name, args))
                        } else {
                            None
                        }
                    }
                    CommentKind::UiTest(line) => {
                        if let Some(args) = line
                            .strip_prefix($ui_test_name)
                            .and_then(|with_colon| with_colon.strip_prefix(':'))
                        {
                            debug!("ui_test {}: {}", $ui_test_name, args);
                            Some(UnexpandedNameValue::new($ui_test_name, args))
                        } else {
                            None
                        }
                    }
                }
            }
        }
        impl NameDirective for $item {
            fn parse_name(&self, comment: &TestComment<'_>) -> bool {
                match comment.comment() {
                    CommentKind::Compiletest(line) => {
                        line.strip_prefix($compiletest_name).is_some_and(|rest| {
                            matches!(rest.chars().next(), None | Some(' ') | Some(':'))
                        })
                    }
                    CommentKind::UiTest(line) => {
                        line.strip_prefix($ui_test_name).is_some_and(|rest| {
                            matches!(rest.chars().next(), None | Some(' ') | Some(':'))
                        })
                    }
                }
            }
            fn parse_name_negative(&self, comment: &TestComment<'_>) -> bool {
                match comment.comment() {
                    CommentKind::Compiletest(line) => {
                        line.strip_prefix("no-").is_some_and(|positive| {
                            positive.strip_prefix($compiletest_name).is_some_and(|rest| {
                                matches!(rest.chars().next(), None | Some(' ') | Some(':'))
                            })
                        })
                    }
                    CommentKind::UiTest(line) => line.strip_prefix("no-").is_some_and(|positive| {
                        positive.strip_prefix($ui_test_name).is_some_and(|rest| {
                            matches!(rest.chars().next(), None | Some(' ') | Some(':'))
                        })
                    }),
                }
            }
        }
    };
}

// ========================================================================
// Macros are in the form (name, compiletest_name, ui_test_name).
// If ui_test_name does not exist, ui_test does not support that directive.
// ========================================================================
name_value_directive!(ErrorPatternDirective, "error-in-other-file", "error-pattern");
name_value_directive!(CompileFlagsDirective, "compile-flags", "compile-flags");
name_value_directive!(RunFlagsDirective, "run-flags"); // UNUSED IN UI TESTS
name_value_directive!(PrettyModeDirective, "pretty-mode"); // UNUSED IN UI TESTS
name_value_directive!(AuxBuildDirective, "aux-build", "aux-build");
name_value_directive!(AuxCrateDirective, "aux-crate"); // UNUSED IN UI TESTS
name_value_directive!(ExecEnvDirective, "exec-env");
name_value_directive!(UnsetExecEnvDirective, "unset-exec-env"); // UNUSED IN UI TESTS
name_value_directive!(RustcEnvDirective, "rustc-env", "rustc-env");
name_value_directive!(UnsetRustcEnvDirective, "unset-rustc-env"); // UNUSED IN UI TESTS
name_value_directive!(ForbidOutputDirective, "forbid-output"); // UNUSED IN UI TESTS
name_value_directive!(FailureStatusDirective, "failure-status"); // FIXME: (ui_test) is this like ui_test's run w/ output code?
name_value_directive!(AssemblyOutputDirective, "assembly-output"); // UNUSED IN UI TESTS
name_value_directive!(MirUnitTestDirective, "unit-test"); // UNUSED IN UI TESTS
name_value_directive!(RevisionsDirective, "revisions", "revisions");
name_value_directive!(EditionDirective, "edition", "edition");
// This one is special and not handled like the other needs-
name_value_directive!(NeedsLlvmComponentsDirective, "needs-llvm-components"); // UNUSED IN UI TESTS

// This is not a real directive! This exists so that the parsing infrastructure can
// detect this common typo and error with a more helpful message.
name_value_directive!(IncorrectCompileFlagsDirective, "compiler-flags", "compiler-flags");

name_directive!(ShouldIceDirective, "should-ice"); // UNUSED IN UI TESTS (maybe we want it though?)
name_directive!(BuildAuxDocsDirective, "build-aux-docs"); // UNUSED IN UI TESTS
name_directive!(ForceHostDirective, "force-host"); // UNUSED IN UI TESTS
name_directive!(CheckStdoutDirective, "check-stdout"); // UNUSED IN UI TESTS
name_directive!(CheckRunResultsDirective, "check-run-results"); // UNUSED IN UI TESTS
name_directive!(DontCheckCompilerStdoutDirective, "dont-check-compiler-stdout");
name_directive!(DontCheckCompilerStderrDirective, "dont-check-compiler-stderr");
name_directive!(NoPreferDynamicDirective, "no-prefer-dynamic");
name_directive!(PrettyExpandedDirective, "pretty-expanded");
name_directive!(PrettyCompareOnlyDirective, "pretty-compare-only"); // UNUSED IN UI TESTS
name_directive!(CheckTestLineNumbersMatchDirective, "check-test-line-numbers-match"); // UNUSED IN UI TESTS
name_directive!(IgnorePassDirective, "ignore-pass");
name_directive!(DontCheckFailureStatusDirective, "dont-check-failure-status"); // UNUSED IN UI TESTS
name_directive!(RunRustfixDirective, "run-rustfix", "run-rustfix");
name_directive!(RustfixOnlyMachineApplicableDirective, "rustfix-only-machine-applicable");
name_directive!(StderrPerBitwidthDirective, "stderr-per-bitwidth", "stderr-per-bitwidth");
name_directive!(IncrementalDirective, "incremental"); // UNUSED IN UI TESTS
name_directive!(RemapSrcBaseDirective, "remap-src-base"); // UNUSED IN UI TESTS
name_directive!(CompareOutputLinesBySubsetDirective, "compare-output-lines-by-subset"); // UNUSED IN UI TESTS

// FIXME: (ui_test) ui_test doesn't have granular enough comments for the various check/build/run modes
name_directive!(CheckPassDirective, "check-pass", "check-pass");
name_directive!(BuildPassDirective, "build-pass");
name_directive!(RunPassDirective, "run-pass", "run");

name_directive!(CompileFailDirective, "compile-fail"); // UNUSED IN UI TESTS
name_directive!(CheckFailDirective, "check-fail");
name_directive!(BuildFailDirective, "build-fail");
name_directive!(RunFailDirective, "run-fail"); // UNUSED IN UI TESTS

name_directive!(ShouldFailDirective, "should-fail");

name_val_or_name_directive!(KnownBugDirective, "known-bug"); // UNUSED IN UI TESTS
name_val_or_name_directive!(PPExactDirective, "pp-exact"); // UNUSED IN UI TESTS
