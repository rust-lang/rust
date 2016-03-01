use rustc::lint::*;
use syntax::codemap::Span;
use syntax::parse::token::InternedString;
use syntax::ast::*;
use syntax::visit::{self, FnKind};
use utils::{span_note_and_lint, in_macro};
use strsim::levenshtein;

/// **What it does:** This lint warns about names that are very similar and thus confusing
///
/// **Why is this bad?** It's hard to distinguish between names that differ only by a single character
///
/// **Known problems:** None?
///
/// **Example:** `checked_exp` and `checked_expr`
declare_lint! {
    pub SIMILAR_NAMES,
    Warn,
    "similarly named items and bindings"
}

pub struct SimilarNames(pub usize);

impl LintPass for SimilarNames {
    fn get_lints(&self) -> LintArray {
        lint_array!(SIMILAR_NAMES)
    }
}

struct SimilarNamesLocalVisitor<'a, 'b: 'a> {
    names: Vec<(InternedString, Span)>,
    cx: &'a EarlyContext<'b>,
    limit: usize,
}

const WHITELIST: &'static [&'static str] = &[
    "lhs", "rhs",
];

struct SimilarNamesNameVisitor<'a, 'b: 'a, 'c: 'b>(&'a mut SimilarNamesLocalVisitor<'b, 'c>);

impl<'v, 'a, 'b, 'c> visit::Visitor<'v> for SimilarNamesNameVisitor<'a, 'b, 'c> {
    fn visit_pat(&mut self, pat: &'v Pat) {
        if let PatKind::Ident(_, id, _) = pat.node {
            self.check_name(id.span, id.node.name);
        }
        visit::walk_pat(self, pat);
    }
}

impl<'a, 'b, 'c> SimilarNamesNameVisitor<'a, 'b, 'c> {
    fn check_name(&mut self, span: Span, name: Name) {
        if in_macro(self.0.cx, span) {
            return;
        }
        let interned_name = name.as_str();
        if interned_name.chars().any(char::is_uppercase) {
            return;
        }
        if interned_name.chars().count() < 3 {
            return;
        }
        for &allow in WHITELIST {
            if interned_name == allow {
                return;
            }
            if interned_name.len() <= allow.len() {
                continue;
            }
            // allow_*
            let allow_start = allow.chars().chain(Some('_'));
            if interned_name.chars().zip(allow_start).all(|(l, r)| l == r) {
                return;
            }
            // *_allow
            let allow_end = Some('_').into_iter().chain(allow.chars());
            if interned_name.chars().rev().zip(allow_end.rev()).all(|(l, r)| l == r) {
                return;
            }
        }
        for &(ref existing_name, sp) in &self.0.names {
            let dist = levenshtein(&interned_name, &existing_name);
            // equality is caught by shadow lints
            if dist == 0 {
                continue;
            }
            // if they differ enough it's all good
            if dist > self.0.limit {
                continue;
            }
            // are we doing stuff like `for item in items`?
            if interned_name.starts_with(&**existing_name) ||
               existing_name.starts_with(&*interned_name) ||
               interned_name.ends_with(&**existing_name) ||
               existing_name.ends_with(&*interned_name) {
                continue;
            }
            if dist == 1 {
                // are we doing stuff like a_bar, b_bar, c_bar?
                if interned_name.chars().next() != existing_name.chars().next() && interned_name.chars().nth(1) == Some('_') {
                    continue;
                }
                // are we doing stuff like foo_x, foo_y, foo_z?
                if interned_name.chars().rev().next() != existing_name.chars().rev().next() && interned_name.chars().rev().nth(1) == Some('_') {
                    continue;
                }
            }
            span_note_and_lint(self.0.cx, SIMILAR_NAMES, span, "binding's name is too similar to existing binding", sp, "existing binding defined here");
            return;
        }
        self.0.names.push((interned_name, span));
    }
}

impl<'v, 'a, 'b> visit::Visitor<'v> for SimilarNamesLocalVisitor<'a, 'b> {
    fn visit_local(&mut self, local: &'v Local) {
        SimilarNamesNameVisitor(self).visit_local(local)
    }
    fn visit_block(&mut self, blk: &'v Block) {
        // ensure scoping rules work
        let n = self.names.len();
        visit::walk_block(self, blk);
        self.names.truncate(n);
    }
    fn visit_arm(&mut self, arm: &'v Arm) {
        let n = self.names.len();
        // just go through the first pattern, as either all patterns bind the same bindings or rustc would have errored much earlier
        SimilarNamesNameVisitor(self).visit_pat(&arm.pats[0]);
        self.names.truncate(n);
    }
    fn visit_item(&mut self, _: &'v Item) {
        // do nothing
    }
}

impl EarlyLintPass for SimilarNames {
    fn check_fn(&mut self, cx: &EarlyContext, _: FnKind, decl: &FnDecl, blk: &Block, _: Span, _: NodeId) {
        let mut visitor = SimilarNamesLocalVisitor {
            names: Vec::new(),
            cx: cx,
            limit: self.0,
        };
        // initialize with function arguments
        for arg in &decl.inputs {
            visit::walk_pat(&mut SimilarNamesNameVisitor(&mut visitor), &arg.pat);
        }
        // walk all other bindings
        visit::walk_block(&mut visitor, blk);
    }
}
