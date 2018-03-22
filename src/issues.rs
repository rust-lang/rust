// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Objects for seeking through a char stream for occurrences of TODO and FIXME.
// Depending on the loaded configuration, may also check that these have an
// associated issue number.

use std::fmt;

use config::ReportTactic;

const TO_DO_CHARS: &[char] = &['t', 'o', 'd', 'o'];
const FIX_ME_CHARS: &[char] = &['f', 'i', 'x', 'm', 'e'];

// Enabled implementation detail is here because it is
// irrelevant outside the issues module
fn is_enabled(report_tactic: ReportTactic) -> bool {
    report_tactic != ReportTactic::Never
}

#[derive(Clone, Copy)]
enum Seeking {
    Issue { todo_idx: usize, fixme_idx: usize },
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
    issue_type: IssueType,
    // Indicates whether we're looking for issues with missing numbers, or
    // all issues of this type.
    missing_number: bool,
}

impl fmt::Display for Issue {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        let msg = match self.issue_type {
            IssueType::Todo => "TODO",
            IssueType::Fixme => "FIXME",
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
enum IssueType {
    Todo,
    Fixme,
}

enum IssueClassification {
    Good,
    Bad(Issue),
    None,
}

pub struct BadIssueSeeker {
    state: Seeking,
    report_todo: ReportTactic,
    report_fixme: ReportTactic,
}

impl BadIssueSeeker {
    pub fn new(report_todo: ReportTactic, report_fixme: ReportTactic) -> BadIssueSeeker {
        BadIssueSeeker {
            state: Seeking::Issue {
                todo_idx: 0,
                fixme_idx: 0,
            },
            report_todo,
            report_fixme,
        }
    }

    pub fn is_disabled(&self) -> bool {
        !is_enabled(self.report_todo) && !is_enabled(self.report_fixme)
    }

    // Check whether or not the current char is conclusive evidence for an
    // unnumbered TO-DO or FIX-ME.
    pub fn inspect(&mut self, c: char) -> Option<Issue> {
        match self.state {
            Seeking::Issue {
                todo_idx,
                fixme_idx,
            } => {
                self.state = self.inspect_issue(c, todo_idx, fixme_idx);
            }
            Seeking::Number { issue, part } => {
                let result = self.inspect_number(c, issue, part);

                if let IssueClassification::None = result {
                    return None;
                }

                self.state = Seeking::Issue {
                    todo_idx: 0,
                    fixme_idx: 0,
                };

                if let IssueClassification::Bad(issue) = result {
                    return Some(issue);
                }
            }
        }

        None
    }

    fn inspect_issue(&mut self, c: char, mut todo_idx: usize, mut fixme_idx: usize) -> Seeking {
        if let Some(lower_case_c) = c.to_lowercase().next() {
            if is_enabled(self.report_todo) && lower_case_c == TO_DO_CHARS[todo_idx] {
                todo_idx += 1;
                if todo_idx == TO_DO_CHARS.len() {
                    return Seeking::Number {
                        issue: Issue {
                            issue_type: IssueType::Todo,
                            missing_number: if let ReportTactic::Unnumbered = self.report_todo {
                                true
                            } else {
                                false
                            },
                        },
                        part: NumberPart::OpenParen,
                    };
                }
                fixme_idx = 0;
            } else if is_enabled(self.report_fixme) && lower_case_c == FIX_ME_CHARS[fixme_idx] {
                // Exploit the fact that the character sets of todo and fixme
                // are disjoint by adding else.
                fixme_idx += 1;
                if fixme_idx == FIX_ME_CHARS.len() {
                    return Seeking::Number {
                        issue: Issue {
                            issue_type: IssueType::Fixme,
                            missing_number: if let ReportTactic::Unnumbered = self.report_fixme {
                                true
                            } else {
                                false
                            },
                        },
                        part: NumberPart::OpenParen,
                    };
                }
                todo_idx = 0;
            } else {
                todo_idx = 0;
                fixme_idx = 0;
            }
        }

        Seeking::Issue {
            todo_idx,
            fixme_idx,
        }
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
                if c >= '0' && c <= '9' {
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
        let mut seeker = BadIssueSeeker::new(ReportTactic::Unnumbered, ReportTactic::Unnumbered);
        assert_eq!(
            Some(failing_pos),
            text.find(|c| seeker.inspect(c).is_some())
        );
    }

    fn check_pass(text: &str) {
        let mut seeker = BadIssueSeeker::new(ReportTactic::Unnumbered, ReportTactic::Unnumbered);
        assert_eq!(None, text.find(|c| seeker.inspect(c).is_some()));
    }

    check_fail("TODO\n", 4);
    check_pass(" TO FIX DOME\n");
    check_fail(" \n FIXME\n", 8);
    check_fail("FIXME(\n", 6);
    check_fail("FIXME(#\n", 7);
    check_fail("FIXME(#1\n", 8);
    check_fail("FIXME(#)1\n", 7);
    check_pass("FIXME(#1222)\n");
    check_fail("FIXME(#12\n22)\n", 9);
    check_pass("FIXME(@maintainer, #1222, hello)\n");
    check_fail("TODO(#22) FIXME\n", 15);
}

#[test]
fn find_issue() {
    fn is_bad_issue(text: &str, report_todo: ReportTactic, report_fixme: ReportTactic) -> bool {
        let mut seeker = BadIssueSeeker::new(report_todo, report_fixme);
        text.chars().any(|c| seeker.inspect(c).is_some())
    }

    assert!(is_bad_issue(
        "TODO(@maintainer, #1222, hello)\n",
        ReportTactic::Always,
        ReportTactic::Never,
    ));

    assert!(!is_bad_issue(
        "TODO: no number\n",
        ReportTactic::Never,
        ReportTactic::Always,
    ));

    assert!(!is_bad_issue(
        "Todo: mixed case\n",
        ReportTactic::Never,
        ReportTactic::Always,
    ));

    assert!(is_bad_issue(
        "This is a FIXME(#1)\n",
        ReportTactic::Never,
        ReportTactic::Always,
    ));

    assert!(is_bad_issue(
        "This is a FixMe(#1) mixed case\n",
        ReportTactic::Never,
        ReportTactic::Always,
    ));

    assert!(!is_bad_issue(
        "bad FIXME\n",
        ReportTactic::Always,
        ReportTactic::Never,
    ));
}

#[test]
fn issue_type() {
    let mut seeker = BadIssueSeeker::new(ReportTactic::Always, ReportTactic::Never);
    let expected = Some(Issue {
        issue_type: IssueType::Todo,
        missing_number: false,
    });

    assert_eq!(
        expected,
        "TODO(#100): more awesomeness"
            .chars()
            .map(|c| seeker.inspect(c))
            .find(Option::is_some)
            .unwrap()
    );

    let mut seeker = BadIssueSeeker::new(ReportTactic::Never, ReportTactic::Unnumbered);
    let expected = Some(Issue {
        issue_type: IssueType::Fixme,
        missing_number: true,
    });

    assert_eq!(
        expected,
        "Test. FIXME: bad, bad, not good"
            .chars()
            .map(|c| seeker.inspect(c))
            .find(Option::is_some)
            .unwrap()
    );
}
