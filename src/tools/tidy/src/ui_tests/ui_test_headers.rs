use std::fs::File;
use std::io::{self, BufReader};
use std::path::Path;
use test_common::directives::*;
use test_common::{CommentKind, TestComment};

const KNOWN_DIRECTIVES: &[&dyn TestDirective] = [
    &ErrorPatternDirective as _,
    &CompileFlagsDirective as _,
    &AuxBuildDirective as _,
    &RustcEnvDirective as _,
    &RevisionsDirective as _,
    &EditionDirective as _,
    &RunRustfixDirective as _,
    &StderrPerBitwidthDirective as _,
    &CheckPassDirective as _,
    &RunPassDirective as _,
    // FIXME (ui_test): needs-asm-support
]
.as_slice();

/// Check that a file uses ui_test headers if a ui_test version of a header exists.
pub(super) fn check_file_headers(file_path: &Path) -> Result<(), HeaderError> {
    let f = File::open(file_path)?;
    let rdr = BufReader::new(f);

    let mut errors = vec![];

    test_common::iter_header(file_path, rdr, &mut |comment| {
        let line_num = comment.line_num();
        for &directive in KNOWN_DIRECTIVES {
            let directive_match = match_comment(comment, directive);
            // Only one directive will ever match a line, so any path that matches should break
            match directive_match {
                DirectiveMatchResult::NoMatch => {}
                DirectiveMatchResult::NoActionNeeded => {
                    break;
                }
                DirectiveMatchResult::UseUiTestComment => {
                    errors.push(HeaderAction {
                        line_num,
                        line: comment.full_line().to_string(),
                        action: LineAction::UseUiTestComment,
                    });
                    break;
                }
                DirectiveMatchResult::MigrateToUiTest => {
                    errors.push(HeaderAction {
                        line_num,
                        line: comment.full_line().to_string(),
                        action: LineAction::MigrateToUiTest {
                            compiletest_name: directive.compiletest_name(),
                            ui_test_name: directive.ui_test_name().unwrap(),
                        },
                    });
                    break;
                }
                DirectiveMatchResult::UseUITestName => {
                    errors.push(HeaderAction {
                        line_num,
                        line: comment.full_line().to_string(),
                        action: LineAction::UseUITestName {
                            compiletest_name: directive.compiletest_name(),
                            ui_test_name: directive.ui_test_name().unwrap(),
                        },
                    });
                    break;
                }
            }
        }
    });

    if errors.len() > 0 {
        return Err(HeaderError::InvalidHeader { bad_lines: errors });
    }

    Ok(())
}

fn match_comment(comment: TestComment<'_>, directive: &dyn TestDirective) -> DirectiveMatchResult {
    // See the comments on DirectiveMatchResult variants for more information.
    match comment.comment() {
        CommentKind::Compiletest(line) => {
            if line.starts_with(directive.ui_test_name().unwrap())
                && matches!(
                    line.get(
                        directive.ui_test_name().unwrap().len()
                            ..directive.ui_test_name().unwrap().len() + 1
                    ),
                    Some(":" | "\n")
                )
            {
                DirectiveMatchResult::UseUiTestComment
            } else if line.starts_with(directive.compiletest_name())
                && matches!(
                    line.get(
                        directive.compiletest_name().len()..directive.compiletest_name().len() + 1
                    ),
                    Some(":" | "\n")
                )
            {
                DirectiveMatchResult::MigrateToUiTest
            } else {
                DirectiveMatchResult::NoMatch
            }
        }
        CommentKind::UiTest(line) => {
            if line.starts_with(directive.ui_test_name().unwrap())
                && matches!(
                    line.get(
                        directive.ui_test_name().unwrap().len()
                            ..directive.ui_test_name().unwrap().len() + 1
                    ),
                    Some(":" | "\n")
                )
            {
                DirectiveMatchResult::NoActionNeeded
            } else if line.starts_with(directive.compiletest_name())
                && matches!(
                    line.get(
                        directive.compiletest_name().len()..directive.compiletest_name().len() + 1
                    ),
                    Some(":" | "\n")
                )
            {
                DirectiveMatchResult::UseUITestName
            } else {
                DirectiveMatchResult::NoMatch
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum DirectiveMatchResult {
    /// The directive did not match this comment
    NoMatch,
    /// The directive is known to ui_test and has the correct name. No action
    /// is needed.
    NoActionNeeded,
    /// The directive was a compiletest comment, but it has the right name for
    /// ui_test. It should migrate the comment type without changing the name.
    UseUiTestComment,
    /// The directive was a compiletest comment and should be migrated to a ui_test comment.
    MigrateToUiTest,
    /// The directive was a ui_test style directive, but it was using the compiletest style name.
    /// It must change its name.
    UseUITestName,
}

#[derive(Debug)]
pub(super) struct HeaderAction {
    line_num: usize,
    line: String,
    action: LineAction,
}

impl HeaderAction {
    pub const fn line_num(&self) -> usize {
        self.line_num
    }

    pub fn line(&self) -> &str {
        self.line.as_str()
    }

    pub const fn action(&self) -> LineAction {
        self.action
    }

    /// A message of the required action, to be used in diagnostics.
    pub fn error_message(&self) -> String {
        match self.action {
            LineAction::UseUiTestComment => String::from("use a ui_test style //@ comment"),
            LineAction::MigrateToUiTest { ui_test_name, compiletest_name } => {
                format!(
                    "use a ui_test style //@ comment and use the updated name {} instead of {}",
                    ui_test_name, compiletest_name
                )
            }
            LineAction::UseUITestName { compiletest_name, ui_test_name } => {
                format!("use the the updated name {} instead of {}", ui_test_name, compiletest_name)
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) enum LineAction {
    /// The directive was a compiletest comment, but it has the right name for ui_test. It should
    /// migrate the comment type without changing the name.
    UseUiTestComment,
    /// The directive was a compiletest comment and should be migrated to a ui_test comment using
    /// the name specified.
    MigrateToUiTest { compiletest_name: &'static str, ui_test_name: &'static str },
    /// The directive was a ui_test style directive, but it was using the compiletest style name.
    /// It must change its name.
    UseUITestName { compiletest_name: &'static str, ui_test_name: &'static str },
}

#[derive(Debug)]
pub(super) enum HeaderError {
    IoError(io::Error),
    InvalidHeader { bad_lines: Vec<HeaderAction> },
}

impl From<io::Error> for HeaderError {
    fn from(value: io::Error) -> Self {
        Self::IoError(value)
    }
}
