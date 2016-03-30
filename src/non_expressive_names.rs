use rustc::lint::*;
use syntax::codemap::Span;
use syntax::parse::token::InternedString;
use syntax::ast::*;
use syntax::visit::{self, FnKind};
use utils::{span_lint_and_then, in_macro, span_lint};

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

/// **What it does:** This lint warns about having too many variables whose name consists of a single character
///
/// **Why is this bad?** It's hard to memorize what a variable means without a descriptive name.
///
/// **Known problems:** None?
///
/// **Example:** let (a, b, c, d, e, f, g) = (...);
declare_lint! {
    pub MANY_SINGLE_CHAR_NAMES,
    Warn,
    "too many single character bindings"
}

pub struct NonExpressiveNames {
    pub max_single_char_names: u64,
}

impl LintPass for NonExpressiveNames {
    fn get_lints(&self) -> LintArray {
        lint_array!(SIMILAR_NAMES, MANY_SINGLE_CHAR_NAMES)
    }
}

struct SimilarNamesLocalVisitor<'a, 'b: 'a> {
    names: Vec<(InternedString, Span, usize)>,
    cx: &'a EarlyContext<'b>,
    lint: &'a NonExpressiveNames,
    single_char_names: Vec<char>,
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

fn whitelisted(interned_name: &str) -> bool {
    for &allow in WHITELIST {
        if interned_name == allow {
            return true;
        }
        if interned_name.len() <= allow.len() {
            continue;
        }
        // allow_*
        let allow_start = allow.chars().chain(Some('_'));
        if interned_name.chars().zip(allow_start).all(|(l, r)| l == r) {
            return true;
        }
        // *_allow
        let allow_end = Some('_').into_iter().chain(allow.chars());
        if interned_name.chars().rev().zip(allow_end.rev()).all(|(l, r)| l == r) {
            return true;
        }
    }
    false
}

impl<'a, 'b, 'c> SimilarNamesNameVisitor<'a, 'b, 'c> {
    fn check_short_name(&mut self, c: char, span: Span) {
        // make sure we ignore shadowing
        if self.0.single_char_names.contains(&c) {
            return;
        }
        self.0.single_char_names.push(c);
        if self.0.single_char_names.len() as u64 >= self.0.lint.max_single_char_names {
            span_lint(self.0.cx,
                      MANY_SINGLE_CHAR_NAMES,
                      span,
                      &format!("{}th binding whose name is just one char",
                               self.0.single_char_names.len()));
        }
    }
    fn check_name(&mut self, span: Span, name: Name) {
        if in_macro(self.0.cx, span) {
            return;
        }
        let interned_name = name.as_str();
        if interned_name.chars().any(char::is_uppercase) {
            return;
        }
        let count = interned_name.chars().count();
        if count < 3 {
            if count != 1 {
                return;
            }
            let c = interned_name.chars().next().expect("already checked");
            self.check_short_name(c, span);
            return;
        }
        if whitelisted(&interned_name) {
            return;
        }
        for &(ref existing_name, sp, existing_len) in &self.0.names {
            let mut split_at = None;
            if existing_len > count {
                if existing_len - count != 1 {
                    continue;
                }
                if levenstein_not_1(&interned_name, &existing_name) {
                    continue;
                }
            } else if existing_len < count {
                if count - existing_len != 1 {
                    continue;
                }
                if levenstein_not_1(&existing_name, &interned_name) {
                    continue;
                }
            } else {
                let mut interned_chars = interned_name.chars();
                let mut existing_chars = existing_name.chars();

                if interned_chars.next() != existing_chars.next() {
                    let i = interned_chars.next().expect("we know we have more than 1 char");
                    let e = existing_chars.next().expect("we know we have more than 1 char");
                    if i == e {
                        if i == '_' {
                            // allowed similarity x_foo, y_foo
                            // or too many chars differ (x_foo, y_boo)
                            continue;
                        } else if interned_chars.ne(existing_chars) {
                            // too many chars differ
                            continue
                        }
                    } else {
                        // too many chars differ
                        continue;
                    }
                    split_at = interned_name.chars().next().map(|c| c.len_utf8());
                } else if interned_chars.next_back() == existing_chars.next_back() {
                    if interned_chars.zip(existing_chars).filter(|&(i, e)| i != e).count() != 1 {
                        // too many chars differ, or none differ (aka shadowing)
                        continue;
                    }
                } else {
                    let i = interned_chars.next_back().expect("we know we have more than 2 chars");
                    let e = existing_chars.next_back().expect("we know we have more than 2 chars");
                    if i == e {
                        if i == '_' {
                            // allowed similarity foo_x, foo_x
                            // or too many chars differ (foo_x, boo_x)
                            continue;
                        } else if interned_chars.ne(existing_chars) {
                            // too many chars differ
                            continue
                        }
                    } else {
                        // too many chars differ
                        continue;
                    }
                    split_at = interned_name.char_indices().rev().next().map(|(i, _)| i);
                }
            }
            span_lint_and_then(self.0.cx,
                               SIMILAR_NAMES,
                               span,
                               "binding's name is too similar to existing binding",
                               |diag| {
                                   diag.span_note(sp, "existing binding defined here");
                                   if let Some(split) = split_at {
                                       diag.span_help(span, &format!("separate the discriminating character \
                                                                      by an underscore like: `{}_{}`",
                                                                     &interned_name[..split],
                                                                     &interned_name[split..]));
                                   }
                               });
            return;
        }
        self.0.names.push((interned_name, span, count));
    }
}

impl<'a, 'b> SimilarNamesLocalVisitor<'a, 'b> {
    /// ensure scoping rules work
    fn apply<F: for<'c> Fn(&'c mut Self)>(&mut self, f: F) {
        let n = self.names.len();
        let single_char_count = self.single_char_names.len();
        f(self);
        self.names.truncate(n);
        self.single_char_names.truncate(single_char_count);
    }
}

impl<'v, 'a, 'b> visit::Visitor<'v> for SimilarNamesLocalVisitor<'a, 'b> {
    fn visit_local(&mut self, local: &'v Local) {
        if let Some(ref init) = local.init {
            self.apply(|this| visit::walk_expr(this, &**init));
        }
        // add the pattern after the expression because the bindings aren't available yet in the init expression
        SimilarNamesNameVisitor(self).visit_pat(&*local.pat);
    }
    fn visit_block(&mut self, blk: &'v Block) {
        self.apply(|this| visit::walk_block(this, blk));
    }
    fn visit_arm(&mut self, arm: &'v Arm) {
        self.apply(|this| {
            // just go through the first pattern, as either all patterns bind the same bindings or rustc would have errored much earlier
            SimilarNamesNameVisitor(this).visit_pat(&arm.pats[0]);
            this.apply(|this| visit::walk_expr(this, &arm.body));
        });
    }
    fn visit_item(&mut self, _: &'v Item) {
        // do nothing
    }
}

impl EarlyLintPass for NonExpressiveNames {
    fn check_fn(&mut self, cx: &EarlyContext, _: FnKind, decl: &FnDecl, blk: &Block, _: Span, _: NodeId) {
        let mut visitor = SimilarNamesLocalVisitor {
            names: Vec::new(),
            cx: cx,
            lint: &self,
            single_char_names: Vec::new(),
        };
        // initialize with function arguments
        for arg in &decl.inputs {
            visit::walk_pat(&mut SimilarNamesNameVisitor(&mut visitor), &arg.pat);
        }
        // walk all other bindings
        visit::walk_block(&mut visitor, blk);
    }
}

/// Precondition: `a_name.chars().count() < b_name.chars().count()`.
fn levenstein_not_1(a_name: &str, b_name: &str) -> bool {
    debug_assert!(a_name.chars().count() < b_name.chars().count());
    let mut a_chars = a_name.chars();
    let mut b_chars = b_name.chars();
    while let (Some(a), Some(b)) = (a_chars.next(), b_chars.next()) {
        if a == b {
            continue;
        }
        if let Some(b2) = b_chars.next() {
            // check if there's just one character inserted
            return a != b2 || a_chars.ne(b_chars);
        } else {
            // tuple
            // ntuple
            return true;
        }
    }
    // for item in items
    true
}
