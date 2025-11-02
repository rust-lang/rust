use clippy_config::Conf;
use clippy_utils::diagnostics::{span_lint, span_lint_and_then};
use rustc_ast::ast::{
    self, Arm, AssocItem, AssocItemKind, Attribute, Block, FnDecl, Item, ItemKind, Local, Pat, PatKind,
};
use rustc_ast::visit::{Visitor, walk_block, walk_expr, walk_pat};
use rustc_lint::{EarlyContext, EarlyLintPass, LintContext};
use rustc_session::impl_lint_pass;
use rustc_span::symbol::{Ident, Symbol};
use rustc_span::{Span, sym};
use std::cmp::Ordering;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for names that are very similar and thus confusing.
    ///
    /// Note: this lint looks for similar names throughout each
    /// scope. To allow it, you need to allow it on the scope
    /// level, not on the name that is reported.
    ///
    /// ### Why is this bad?
    /// It's hard to distinguish between names that differ only
    /// by a single character.
    ///
    /// ### Example
    /// ```ignore
    /// let checked_exp = something;
    /// let checked_expr = something_else;
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub SIMILAR_NAMES,
    pedantic,
    "similarly named items and bindings"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for too many variables whose name consists of a
    /// single character.
    ///
    /// ### Why is this bad?
    /// It's hard to memorize what a variable means without a
    /// descriptive name.
    ///
    /// ### Example
    /// ```ignore
    /// let (a, b, c, d, e, f, g) = (...);
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub MANY_SINGLE_CHAR_NAMES,
    pedantic,
    "too many single character bindings"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks if you have variables whose name consists of just
    /// underscores and digits.
    ///
    /// ### Why is this bad?
    /// It's hard to memorize what a variable means without a
    /// descriptive name.
    ///
    /// ### Example
    /// ```no_run
    /// let _1 = 1;
    /// let ___1 = 1;
    /// let __1___2 = 11;
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub JUST_UNDERSCORES_AND_DIGITS,
    style,
    "unclear name"
}

pub struct NonExpressiveNames {
    pub single_char_binding_names_threshold: u64,
}

impl_lint_pass!(NonExpressiveNames => [SIMILAR_NAMES, MANY_SINGLE_CHAR_NAMES, JUST_UNDERSCORES_AND_DIGITS]);

impl NonExpressiveNames {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            single_char_binding_names_threshold: conf.single_char_binding_names_threshold,
        }
    }
}

struct ExistingName {
    interned: Symbol,
    span: Span,
    len: usize,
    exemptions: &'static [&'static str],
}

struct SimilarNamesLocalVisitor<'a, 'tcx> {
    names: Vec<ExistingName>,
    cx: &'a EarlyContext<'tcx>,
    threshold: u64,

    /// A stack of scopes containing the single-character bindings in each scope.
    single_char_names: Vec<Vec<Ident>>,
}

impl SimilarNamesLocalVisitor<'_, '_> {
    fn check_single_char_names(&self) {
        if self.single_char_names.last().map(Vec::len) == Some(0) {
            return;
        }

        let num_single_char_names = self.single_char_names.iter().flatten().count();
        if num_single_char_names as u64 > self.threshold {
            let span = self
                .single_char_names
                .iter()
                .flatten()
                .map(|ident| ident.span)
                .collect::<Vec<_>>();
            span_lint(
                self.cx,
                MANY_SINGLE_CHAR_NAMES,
                span,
                format!("{num_single_char_names} bindings with single-character names in scope"),
            );
        }
    }
}

// this list contains lists of names that are allowed to be similar
// the assumption is that no name is ever contained in multiple lists.
const ALLOWED_TO_BE_SIMILAR: &[&[&str]] = &[
    &["parsed", "parser"],
    &["lhs", "rhs"],
    &["tx", "rx"],
    &["set", "get"],
    &["args", "arms"],
    &["qpath", "path"],
    &["lit", "lint"],
    &["wparam", "lparam"],
    &["iter", "item"],
];

/// Characters that look visually similar
const SIMILAR_CHARS: &[(char, char)] = &[('l', 'i'), ('l', '1'), ('i', '1'), ('u', 'v')];

/// Return true if two characters are visually similar
fn chars_are_similar(a: char, b: char) -> bool {
    a == b || SIMILAR_CHARS.contains(&(a, b)) || SIMILAR_CHARS.contains(&(b, a))
}

struct SimilarNamesNameVisitor<'a, 'tcx, 'b>(&'b mut SimilarNamesLocalVisitor<'a, 'tcx>);

impl<'tcx> Visitor<'tcx> for SimilarNamesNameVisitor<'_, 'tcx, '_> {
    fn visit_pat(&mut self, pat: &'tcx Pat) {
        match pat.kind {
            PatKind::Ident(_, ident, _) => {
                if !pat.span.from_expansion() {
                    self.check_ident(ident);
                }
            },
            PatKind::Struct(_, _, ref fields, _) => {
                for field in fields {
                    if !field.is_shorthand {
                        self.visit_pat(&field.pat);
                    }
                }
            },
            // just go through the first pattern, as either all patterns
            // bind the same bindings or rustc would have errored much earlier
            PatKind::Or(ref pats) => self.visit_pat(&pats[0]),
            _ => walk_pat(self, pat),
        }
    }
}

#[must_use]
fn get_exemptions(interned_name: &str) -> Option<&'static [&'static str]> {
    ALLOWED_TO_BE_SIMILAR
        .iter()
        .find(|&&list| allowed_to_be_similar(interned_name, list))
        .copied()
}

#[must_use]
fn allowed_to_be_similar(interned_name: &str, list: &[&str]) -> bool {
    list.iter()
        .any(|&name| interned_name.starts_with(name) || interned_name.ends_with(name))
}

impl SimilarNamesNameVisitor<'_, '_, '_> {
    fn check_short_ident(&mut self, ident: Ident) {
        // Ignore shadowing
        if self
            .0
            .single_char_names
            .iter()
            .flatten()
            .any(|id| id.name == ident.name)
        {
            return;
        }

        if let Some(scope) = &mut self.0.single_char_names.last_mut() {
            scope.push(ident);
        }
    }

    fn check_ident(&mut self, ident: Ident) {
        let interned_name = ident.name.as_str();
        // name can be empty if it comes from recovery
        if interned_name.chars().any(char::is_uppercase) || interned_name.is_empty() {
            return;
        }
        if interned_name.chars().all(|c| c.is_ascii_digit() || c == '_') {
            span_lint(
                self.0.cx,
                JUST_UNDERSCORES_AND_DIGITS,
                ident.span,
                "consider choosing a more descriptive name",
            );
            return;
        }
        if interned_name.starts_with('_') {
            // these bindings are typically unused or represent an ignored portion of a destructuring pattern
            return;
        }
        let count = interned_name.chars().count();
        if count < 3 {
            if count == 1 {
                self.check_short_ident(ident);
            }
            return;
        }
        for existing_name in &self.0.names {
            if allowed_to_be_similar(interned_name, existing_name.exemptions) {
                continue;
            }

            let existing_str = existing_name.interned.as_str();

            // The first char being different is usually enough to set identifiers apart, as long
            // as the characters aren't too similar.
            if !chars_are_similar(
                interned_name.chars().next().expect("len >= 1"),
                existing_str.chars().next().expect("len >= 1"),
            ) {
                continue;
            }

            // Skip similarity check if both names are exactly 3 characters
            if count == 3 && existing_name.len == 3 {
                continue;
            }

            let dissimilar = match existing_name.len.cmp(&count) {
                Ordering::Greater => existing_name.len - count != 1 || levenstein_not_1(interned_name, existing_str),
                Ordering::Less => count - existing_name.len != 1 || levenstein_not_1(existing_str, interned_name),
                Ordering::Equal => Self::equal_length_strs_not_similar(interned_name, existing_str),
            };

            if dissimilar {
                continue;
            }

            span_lint_and_then(
                self.0.cx,
                SIMILAR_NAMES,
                ident.span,
                "binding's name is too similar to existing binding",
                |diag| {
                    diag.span_note(existing_name.span, "existing binding defined here");
                },
            );
            return;
        }
        self.0.names.push(ExistingName {
            interned: ident.name,
            span: ident.span,
            len: count,
            exemptions: get_exemptions(interned_name).unwrap_or(&[]),
        });
    }

    fn equal_length_strs_not_similar(interned_name: &str, existing_name: &str) -> bool {
        let mut interned_chars = interned_name.chars();
        let mut existing_chars = existing_name.chars();
        let first_i = interned_chars.next().expect("we know we have at least one char");
        let first_e = existing_chars.next().expect("we know we have at least one char");
        let eq_or_numeric = |(a, b): (char, char)| a == b || a.is_numeric() && b.is_numeric();

        if eq_or_numeric((first_i, first_e)) {
            let last_i = interned_chars.next_back().expect("we know we have at least two chars");
            let last_e = existing_chars.next_back().expect("we know we have at least two chars");
            if eq_or_numeric((last_i, last_e)) {
                if interned_chars
                    .zip(existing_chars)
                    .filter(|&ie| !eq_or_numeric(ie))
                    .count()
                    != 1
                {
                    return true;
                }
            } else {
                let second_last_i = interned_chars
                    .next_back()
                    .expect("we know we have at least three chars");
                let second_last_e = existing_chars
                    .next_back()
                    .expect("we know we have at least three chars");
                if !eq_or_numeric((second_last_i, second_last_e))
                    || second_last_i == '_'
                    || !interned_chars.zip(existing_chars).all(eq_or_numeric)
                {
                    // allowed similarity foo_x, foo_y
                    // or too many chars differ (foo_x, boo_y) or (foox, booy)
                    return true;
                }
            }
        } else {
            let second_i = interned_chars.next().expect("we know we have at least two chars");
            let second_e = existing_chars.next().expect("we know we have at least two chars");
            if !eq_or_numeric((second_i, second_e))
                || second_i == '_'
                || !interned_chars.zip(existing_chars).all(eq_or_numeric)
            {
                // allowed similarity x_foo, y_foo
                // or too many chars differ (x_foo, y_boo) or (xfoo, yboo)
                return true;
            }
        }

        false
    }
}

impl SimilarNamesLocalVisitor<'_, '_> {
    /// ensure scoping rules work
    fn apply<F: for<'c> Fn(&'c mut Self)>(&mut self, f: F) {
        let n = self.names.len();
        let single_char_count = self.single_char_names.len();
        f(self);
        self.names.truncate(n);
        self.single_char_names.truncate(single_char_count);
    }
}

impl<'tcx> Visitor<'tcx> for SimilarNamesLocalVisitor<'_, 'tcx> {
    fn visit_local(&mut self, local: &'tcx Local) {
        if let Some((init, els)) = &local.kind.init_else_opt() {
            self.apply(|this| walk_expr(this, init));
            if let Some(els) = els {
                self.apply(|this| walk_block(this, els));
            }
        }
        // add the pattern after the expression because the bindings aren't available
        // yet in the init
        // expression
        SimilarNamesNameVisitor(self).visit_pat(&local.pat);
    }
    fn visit_block(&mut self, blk: &'tcx Block) {
        self.single_char_names.push(vec![]);

        self.apply(|this| walk_block(this, blk));

        self.check_single_char_names();
        self.single_char_names.pop();
    }
    fn visit_arm(&mut self, arm: &'tcx Arm) {
        self.single_char_names.push(vec![]);

        self.apply(|this| {
            SimilarNamesNameVisitor(this).visit_pat(&arm.pat);
            if let Some(body) = &arm.body {
                this.apply(|this| walk_expr(this, body));
            }
        });

        self.check_single_char_names();
        self.single_char_names.pop();
    }
    fn visit_item(&mut self, _: &Item) {
        // do not recurse into inner items
    }
}

impl EarlyLintPass for NonExpressiveNames {
    fn check_item(&mut self, cx: &EarlyContext<'_>, item: &Item) {
        if item.span.in_external_macro(cx.sess().source_map()) {
            return;
        }

        if let ItemKind::Fn(box ast::Fn {
            ref sig,
            body: Some(ref blk),
            ..
        }) = item.kind
        {
            do_check(self, cx, &item.attrs, &sig.decl, blk);
        }
    }

    fn check_impl_item(&mut self, cx: &EarlyContext<'_>, item: &AssocItem) {
        if item.span.in_external_macro(cx.sess().source_map()) {
            return;
        }

        if let AssocItemKind::Fn(box ast::Fn {
            ref sig,
            body: Some(ref blk),
            ..
        }) = item.kind
        {
            do_check(self, cx, &item.attrs, &sig.decl, blk);
        }
    }
}

fn do_check(lint: &NonExpressiveNames, cx: &EarlyContext<'_>, attrs: &[Attribute], decl: &FnDecl, blk: &Block) {
    if !attrs.iter().any(|attr| attr.has_name(sym::test)) {
        let mut visitor = SimilarNamesLocalVisitor {
            names: Vec::new(),
            cx,
            threshold: lint.single_char_binding_names_threshold,
            single_char_names: vec![vec![]],
        };

        // initialize with function arguments
        for arg in &decl.inputs {
            SimilarNamesNameVisitor(&mut visitor).visit_pat(&arg.pat);
        }
        // walk all other bindings
        walk_block(&mut visitor, blk);

        visitor.check_single_char_names();
    }
}

/// Precondition: `a_name.chars().count() < b_name.chars().count()`.
#[must_use]
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
        }
        // tuple
        // ntuple
        return true;
    }
    // for item in items
    true
}
