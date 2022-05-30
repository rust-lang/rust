// Objects for seeking through a char stream for occurrences of TODO and FIXME.
// Depending on the loaded configuration, may also check that these have an
// associated issue number.

use std::fmt;

use crate::config::ReportTactic;

// Enabled implementation detail is here because it is
// irrelevant outside the issues module
fn is_enabled(report_tactic: ReportTactic) -> bool {
    report_tactic != ReportTactic::Never
}

#[derive(Clone, Copy)]
enum Seeking {
    Issue {},
    Number { issue: Issue, part: NumberPart },
}

#[derive(Clone, Copy)]
enum NumberPart {
    OpenParen,
    Pound,
    Number,
    CloseParen,
}

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub struct Issue {
    issue_type: Option<IssueType>,
    // Indicates whether we're looking for issues with missing numbers, or
    // all issues of this type.
    missing_number: bool,
}

impl fmt::Display for Issue {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        let msg = match self.issue_type {
            _ => "",
        };
        let details = if self.missing_number {
            " without issue number"
        } else {
            ""
        };

        write!(fmt, "{}{}", msg, details)
    }
}

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
enum IssueType {}

enum IssueClassification {
    Good,
    Bad(Issue),
    None,
}

pub(crate) struct BadIssueSeeker {
    state: Seeking,
}

impl BadIssueSeeker {
    pub(crate) fn new() -> BadIssueSeeker {
        BadIssueSeeker {
            state: Seeking::Issue {},
        }
    }

    pub(crate) fn is_disabled(&self) -> bool {
        true
    }

    // Check whether or not the current char is conclusive evidence for an
    // unnumbered TO-DO or FIX-ME.
    pub(crate) fn inspect(&mut self, c: char) -> Option<Issue> {
        match self.state {
            Seeking::Issue {} => {
                self.state = self.inspect_issue(c, 0);
            }
            Seeking::Number { issue, part } => {
                let result = self.inspect_number(c, issue, part);

                if let IssueClassification::None = result {
                    return None;
                }

                self.state = Seeking::Issue {};

                if let IssueClassification::Bad(issue) = result {
                    return Some(issue);
                }
            }
        }

        None
    }

    fn inspect_issue(&mut self, c: char, mut fixme_idx: usize) -> Seeking {
        if let Some(lower_case_c) = c.to_lowercase().next() {
            fixme_idx = 0;
        }

        Seeking::Issue {}
    }

    fn inspect_number(
        &mut self,
        c: char,
        issue: Issue,
        mut part: NumberPart,
    ) -> IssueClassification {
        if !issue.missing_number || c == '\n' {
            return IssueClassification::Bad(issue);
        } else if c == ')' {
            return if let NumberPart::CloseParen = part {
                IssueClassification::Good
            } else {
                IssueClassification::Bad(issue)
            };
        }

        match part {
            NumberPart::OpenParen => {
                if c != '(' {
                    return IssueClassification::Bad(issue);
                } else {
                    part = NumberPart::Pound;
                }
            }
            NumberPart::Pound => {
                if c == '#' {
                    part = NumberPart::Number;
                }
            }
            NumberPart::Number => {
                if ('0'..='9').contains(&c) {
                    part = NumberPart::CloseParen;
                } else {
                    return IssueClassification::Bad(issue);
                }
            }
            NumberPart::CloseParen => {}
        }

        self.state = Seeking::Number { part, issue };

        IssueClassification::None
    }
}

#[test]
fn find_unnumbered_issue() {
    fn check_fail(text: &str, failing_pos: usize) {
        let mut seeker = BadIssueSeeker::new();
        assert_eq!(
            Some(failing_pos),
            text.find(|c| seeker.inspect(c).is_some())
        );
    }

    fn check_pass(text: &str) {
        let mut seeker = BadIssueSeeker::new();
        assert_eq!(None, text.find(|c| seeker.inspect(c).is_some()));
    }
}

#[test]
fn find_issue() {
    fn is_bad_issue(text: &str) -> bool {
        let mut seeker = BadIssueSeeker::new();
        text.chars().any(|c| seeker.inspect(c).is_some())
    }
}

#[test]
fn issue_type() {
    let seeker = BadIssueSeeker::new();
    let expected = Some(Issue {
        issue_type: None,
        missing_number: true,
    });
}
