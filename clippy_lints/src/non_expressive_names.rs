use rustc::lint::*;
use syntax::codemap::Span;
use syntax::parse::token::InternedString;
use syntax::ast::*;
use syntax::attr;
use syntax::visit::{Visitor, walk_block, walk_pat, walk_expr};
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
    Allow,
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

struct ExistingName {
    interned: InternedString,
    span: Span,
    len: usize,
    whitelist: &'static [&'static str],
}

struct SimilarNamesLocalVisitor<'a, 'b: 'a> {
    names: Vec<ExistingName>,
    cx: &'a EarlyContext<'b>,
    lint: &'a NonExpressiveNames,
    single_char_names: Vec<char>,
}

// this list contains lists of names that are allowed to be similar
// the assumption is that no name is ever contained in multiple lists.
#[cfg_attr(rustfmt, rustfmt_skip)]
const WHITELIST: &'static [&'static [&'static str]] = &[
    &["parsed", "parser"],
    &["lhs", "rhs"],
    &["tx", "rx"],
    &["set", "get"],
];

struct SimilarNamesNameVisitor<'a, 'b: 'a, 'c: 'b>(&'a mut SimilarNamesLocalVisitor<'b, 'c>);

impl<'v, 'a, 'b, 'c> Visitor<'v> for SimilarNamesNameVisitor<'a, 'b, 'c> {
    fn visit_pat(&mut self, pat: &'v Pat) {
        match pat.node {
            PatKind::Ident(_, id, _) => self.check_name(id.span, id.node.name),
            PatKind::Struct(_, ref fields, _) => {
                for field in fields {
                    if !field.node.is_shorthand {
                        self.visit_pat(&field.node.pat);
                    }
                }
            }
            _ => walk_pat(self, pat),
        }
    }
}

fn get_whitelist(interned_name: &str) -> Option<&'static [&'static str]> {
    for &allow in WHITELIST {
        if whitelisted(interned_name, allow) {
            return Some(allow);
        }
    }
    None
}

fn whitelisted(interned_name: &str, list: &[&str]) -> bool {
    if list.iter().any(|&name| interned_name == name) {
        return true;
    }
    for name in list {
        // name_*
        if interned_name.chars().zip(name.chars()).all(|(l, r)| l == r) {
            return true;
        }
        // *_name
        if interned_name.chars().rev().zip(name.chars().rev()).all(|(l, r)| l == r) {
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
                      &format!("{}th binding whose name is just one char", self.0.single_char_names.len()));
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
            if count == 1 {
                let c = interned_name.chars().next().expect("already checked");
                self.check_short_name(c, span);
            }
            return;
        }
        for existing_name in &self.0.names {
            if whitelisted(&interned_name, existing_name.whitelist) {
                continue;
            }
            let mut split_at = None;
            if existing_name.len > count {
                if existing_name.len - count != 1 || levenstein_not_1(&interned_name, &existing_name.interned) {
                    continue;
                }
            } else if existing_name.len < count {
                if count - existing_name.len != 1 || levenstein_not_1(&existing_name.interned, &interned_name) {
                    continue;
                }
            } else {
                let mut interned_chars = interned_name.chars();
                let mut existing_chars = existing_name.interned.chars();
                let first_i = interned_chars.next().expect("we know we have at least one char");
                let first_e = existing_chars.next().expect("we know we have at least one char");
                let eq_or_numeric = |a: char, b: char| a == b || a.is_numeric() && b.is_numeric();

                if eq_or_numeric(first_i, first_e) {
                    let last_i = interned_chars.next_back().expect("we know we have at least two chars");
                    let last_e = existing_chars.next_back().expect("we know we have at least two chars");
                    if eq_or_numeric(last_i, last_e) {
                        if interned_chars.zip(existing_chars).filter(|&(i, e)| !eq_or_numeric(i, e)).count() != 1 {
                            continue;
                        }
                    } else {
                        let second_last_i = interned_chars.next_back().expect("we know we have at least three chars");
                        let second_last_e = existing_chars.next_back().expect("we know we have at least three chars");
                        if !eq_or_numeric(second_last_i, second_last_e) || second_last_i == '_' ||
                           !interned_chars.zip(existing_chars).all(|(i, e)| eq_or_numeric(i, e)) {
                            // allowed similarity foo_x, foo_y
                            // or too many chars differ (foo_x, boo_y) or (foox, booy)
                            continue;
                        }
                        split_at = interned_name.char_indices().rev().next().map(|(i, _)| i);
                    }
                } else {
                    let second_i = interned_chars.next().expect("we know we have at least two chars");
                    let second_e = existing_chars.next().expect("we know we have at least two chars");
                    if !eq_or_numeric(second_i, second_e) || second_i == '_' ||
                       !interned_chars.zip(existing_chars).all(|(i, e)| eq_or_numeric(i, e)) {
                        // allowed similarity x_foo, y_foo
                        // or too many chars differ (x_foo, y_boo) or (xfoo, yboo)
                        continue;
                    }
                    split_at = interned_name.chars().next().map(|c| c.len_utf8());
                }
            }
            span_lint_and_then(self.0.cx,
                               SIMILAR_NAMES,
                               span,
                               "binding's name is too similar to existing binding",
                               |diag| {
                diag.span_note(existing_name.span, "existing binding defined here");
                if let Some(split) = split_at {
                    diag.span_help(span,
                                   &format!("separate the discriminating character by an \
                                                                underscore like: `{}_{}`",
                                                               &interned_name[..split],
                                                               &interned_name[split..]));
                }
            });
            return;
        }
        self.0.names.push(ExistingName {
            whitelist: get_whitelist(&interned_name).unwrap_or(&[]),
            interned: interned_name,
            span: span,
            len: count,
        });
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

impl<'v, 'a, 'b> Visitor<'v> for SimilarNamesLocalVisitor<'a, 'b> {
    fn visit_local(&mut self, local: &'v Local) {
        if let Some(ref init) = local.init {
            self.apply(|this| walk_expr(this, &**init));
        }
        // add the pattern after the expression because the bindings aren't available yet in the init expression
        SimilarNamesNameVisitor(self).visit_pat(&*local.pat);
    }
    fn visit_block(&mut self, blk: &'v Block) {
        self.apply(|this| walk_block(this, blk));
    }
    fn visit_arm(&mut self, arm: &'v Arm) {
        self.apply(|this| {
            // just go through the first pattern, as either all patterns bind the same bindings or rustc would have errored much earlier
            SimilarNamesNameVisitor(this).visit_pat(&arm.pats[0]);
            this.apply(|this| walk_expr(this, &arm.body));
        });
    }
    fn visit_item(&mut self, _: &'v Item) {
        // do not recurse into inner items
    }
}

impl EarlyLintPass for NonExpressiveNames {
    fn check_item(&mut self, cx: &EarlyContext, item: &Item) {
        if let ItemKind::Fn(ref decl, _, _, _, _, ref blk) = item.node {
            if !attr::contains_name(&item.attrs, "test") {
                let mut visitor = SimilarNamesLocalVisitor {
                    names: Vec::new(),
                    cx: cx,
                    lint: self,
                    single_char_names: Vec::new(),
                };
                // initialize with function arguments
                for arg in &decl.inputs {
                    SimilarNamesNameVisitor(&mut visitor).visit_pat(&arg.pat);
                }
                // walk all other bindings
                walk_block(&mut visitor, blk);
            }
        }
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
