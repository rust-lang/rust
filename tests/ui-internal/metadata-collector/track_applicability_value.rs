#![deny(clippy::internal)]
#![feature(rustc_private)]

extern crate rustc_ast;
extern crate rustc_errors;
extern crate rustc_lint;
extern crate rustc_session;
extern crate rustc_span;

use rustc_ast::ast::Expr;
use rustc_errors::{Applicability, DiagnosticBuilder};
use rustc_lint::{EarlyContext, EarlyLintPass, Lint, LintContext};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Span;

fn producer_fn() -> Applicability {
    Applicability::MachineApplicable
}

fn modifier_fn(applicability: &mut Applicability) {
    if let Applicability::MaybeIncorrect = applicability {
        *applicability = Applicability::HasPlaceholders;
    }
}

fn consumer_fn(_applicability: Applicability) {}

struct Muh;

impl Muh {
    fn producer_method() -> Applicability {
        Applicability::MachineApplicable
    }
}

fn main() {
    let mut applicability = producer_fn();
    applicability = Applicability::MachineApplicable;
    applicability = Muh::producer_method();

    applicability = if true {
        Applicability::HasPlaceholders
    } else {
        Applicability::MaybeIncorrect
    };

    modifier_fn(&mut applicability);

    consumer_fn(applicability);
}
