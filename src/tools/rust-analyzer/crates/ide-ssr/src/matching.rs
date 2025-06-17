//! This module is responsible for matching a search pattern against a node in the AST. In the
//! process of matching, placeholder values are recorded.

use crate::{
    SsrMatches,
    parsing::{Constraint, NodeKind, Placeholder, Var},
    resolving::{ResolvedPattern, ResolvedRule, UfcsCallInfo},
};
use hir::{FileRange, ImportPathConfig, Semantics};
use ide_db::{FxHashMap, base_db::RootQueryDb};
use std::{cell::Cell, iter::Peekable};
use syntax::{
    SmolStr, SyntaxElement, SyntaxElementChildren, SyntaxKind, SyntaxNode, SyntaxToken,
    ast::{self, AstNode, AstToken, HasGenericArgs},
};

// Creates a match error. If we're currently attempting to match some code that we thought we were
// going to match, as indicated by the --debug-snippet flag, then populate the reason field.
macro_rules! match_error {
    ($e:expr) => {{
            MatchFailed {
                reason: if recording_match_fail_reasons() {
                    Some(format!("{}", $e))
                } else {
                    None
                }
            }
    }};
    ($fmt:expr, $($arg:tt)+) => {{
        MatchFailed {
            reason: if recording_match_fail_reasons() {
                Some(format!($fmt, $($arg)+))
            } else {
                None
            }
        }
    }};
}

// Fails the current match attempt, recording the supplied reason if we're recording match fail reasons.
macro_rules! fail_match {
    ($($args:tt)*) => {return Err(match_error!($($args)*))};
}

/// Information about a match that was found.
#[derive(Debug)]
pub struct Match {
    pub(crate) range: FileRange,
    pub(crate) matched_node: SyntaxNode,
    pub(crate) placeholder_values: FxHashMap<Var, PlaceholderMatch>,
    pub(crate) ignored_comments: Vec<ast::Comment>,
    pub(crate) rule_index: usize,
    /// The depth of matched_node.
    pub(crate) depth: usize,
    // Each path in the template rendered for the module in which the match was found.
    pub(crate) rendered_template_paths: FxHashMap<SyntaxNode, hir::ModPath>,
}

/// Information about a placeholder bound in a match.
#[derive(Debug)]
pub(crate) struct PlaceholderMatch {
    pub(crate) range: FileRange,
    /// More matches, found within `node`.
    pub(crate) inner_matches: SsrMatches,
    /// How many times the code that the placeholder matched needed to be dereferenced. Will only be
    /// non-zero if the placeholder matched to the receiver of a method call.
    pub(crate) autoderef_count: usize,
    pub(crate) autoref_kind: ast::SelfParamKind,
}

#[derive(Debug)]
pub(crate) struct MatchFailureReason {
    pub(crate) reason: String,
}

/// An "error" indicating that matching failed. Use the fail_match! macro to create and return this.
#[derive(Clone)]
pub(crate) struct MatchFailed {
    /// The reason why we failed to match. Only present when debug_active true in call to
    /// `get_match`.
    pub(crate) reason: Option<String>,
}

/// Checks if `code` matches the search pattern found in `search_scope`, returning information about
/// the match, if it does. Since we only do matching in this module and searching is done by the
/// parent module, we don't populate nested matches.
pub(crate) fn get_match<'db>(
    debug_active: bool,
    rule: &ResolvedRule<'db>,
    code: &SyntaxNode,
    restrict_range: &Option<FileRange>,
    sema: &Semantics<'db, ide_db::RootDatabase>,
) -> Result<Match, MatchFailed> {
    record_match_fails_reasons_scope(debug_active, || {
        Matcher::try_match(rule, code, restrict_range, sema)
    })
}

/// Checks if our search pattern matches a particular node of the AST.
struct Matcher<'db, 'sema> {
    sema: &'sema Semantics<'db, ide_db::RootDatabase>,
    /// If any placeholders come from anywhere outside of this range, then the match will be
    /// rejected.
    restrict_range: Option<FileRange>,
    rule: &'sema ResolvedRule<'db>,
}

/// Which phase of matching we're currently performing. We do two phases because most attempted
/// matches will fail and it means we can defer more expensive checks to the second phase.
enum Phase<'a> {
    /// On the first phase, we perform cheap checks. No state is mutated and nothing is recorded.
    First,
    /// On the second phase, we construct the `Match`. Things like what placeholders bind to is
    /// recorded.
    Second(&'a mut Match),
}

impl<'db, 'sema> Matcher<'db, 'sema> {
    fn try_match(
        rule: &ResolvedRule<'db>,
        code: &SyntaxNode,
        restrict_range: &Option<FileRange>,
        sema: &'sema Semantics<'db, ide_db::RootDatabase>,
    ) -> Result<Match, MatchFailed> {
        let match_state = Matcher { sema, restrict_range: *restrict_range, rule };
        // First pass at matching, where we check that node types and idents match.
        match_state.attempt_match_node(&mut Phase::First, &rule.pattern.node, code)?;
        let file_range = sema
            .original_range_opt(code)
            .ok_or(MatchFailed { reason: Some("def site definition".to_owned()) })?;
        match_state.validate_range(&file_range)?;
        let mut the_match = Match {
            range: file_range,
            matched_node: code.clone(),
            placeholder_values: FxHashMap::default(),
            ignored_comments: Vec::new(),
            rule_index: rule.index,
            depth: 0,
            rendered_template_paths: FxHashMap::default(),
        };
        // Second matching pass, where we record placeholder matches, ignored comments and maybe do
        // any other more expensive checks that we didn't want to do on the first pass.
        match_state.attempt_match_node(
            &mut Phase::Second(&mut the_match),
            &rule.pattern.node,
            code,
        )?;
        the_match.depth = sema.ancestors_with_macros(the_match.matched_node.clone()).count();
        if let Some(template) = &rule.template {
            the_match.render_template_paths(template, sema)?;
        }
        Ok(the_match)
    }

    /// Checks that `range` is within the permitted range if any. This is applicable when we're
    /// processing a macro expansion and we want to fail the match if we're working with a node that
    /// didn't originate from the token tree of the macro call.
    fn validate_range(&self, range: &FileRange) -> Result<(), MatchFailed> {
        if let Some(restrict_range) = &self.restrict_range {
            if restrict_range.file_id != range.file_id
                || !restrict_range.range.contains_range(range.range)
            {
                fail_match!("Node originated from a macro");
            }
        }
        Ok(())
    }

    fn attempt_match_node(
        &self,
        phase: &mut Phase<'_>,
        pattern: &SyntaxNode,
        code: &SyntaxNode,
    ) -> Result<(), MatchFailed> {
        // Handle placeholders.
        if let Some(placeholder) = self.get_placeholder_for_node(pattern) {
            for constraint in &placeholder.constraints {
                self.check_constraint(constraint, code)?;
            }
            if let Phase::Second(matches_out) = phase {
                let original_range = self
                    .sema
                    .original_range_opt(code)
                    .ok_or(MatchFailed { reason: Some("def site definition".to_owned()) })?;
                // We validated the range for the node when we started the match, so the placeholder
                // probably can't fail range validation, but just to be safe...
                self.validate_range(&original_range)?;
                matches_out.placeholder_values.insert(
                    placeholder.ident.clone(),
                    PlaceholderMatch::from_range(original_range),
                );
            }
            return Ok(());
        }
        // We allow a UFCS call to match a method call, provided they resolve to the same function.
        if let Some(pattern_ufcs) = self.rule.pattern.ufcs_function_calls.get(pattern) {
            if let Some(code) = ast::MethodCallExpr::cast(code.clone()) {
                return self.attempt_match_ufcs_to_method_call(phase, pattern_ufcs, &code);
            }
            if let Some(code) = ast::CallExpr::cast(code.clone()) {
                return self.attempt_match_ufcs_to_ufcs(phase, pattern_ufcs, &code);
            }
        }
        if pattern.kind() != code.kind() {
            fail_match!(
                "Pattern had `{}` ({:?}), code had `{}` ({:?})",
                pattern.text(),
                pattern.kind(),
                code.text(),
                code.kind()
            );
        }
        // Some kinds of nodes have special handling. For everything else, we fall back to default
        // matching.
        match code.kind() {
            SyntaxKind::RECORD_EXPR_FIELD_LIST => {
                self.attempt_match_record_field_list(phase, pattern, code)
            }
            SyntaxKind::TOKEN_TREE => self.attempt_match_token_tree(phase, pattern, code),
            SyntaxKind::PATH => self.attempt_match_path(phase, pattern, code),
            _ => self.attempt_match_node_children(phase, pattern, code),
        }
    }

    fn attempt_match_node_children(
        &self,
        phase: &mut Phase<'_>,
        pattern: &SyntaxNode,
        code: &SyntaxNode,
    ) -> Result<(), MatchFailed> {
        self.attempt_match_sequences(
            phase,
            PatternIterator::new(pattern),
            code.children_with_tokens(),
        )
    }

    fn attempt_match_sequences(
        &self,
        phase: &mut Phase<'_>,
        pattern_it: PatternIterator,
        mut code_it: SyntaxElementChildren,
    ) -> Result<(), MatchFailed> {
        let mut pattern_it = pattern_it.peekable();
        loop {
            match phase.next_non_trivial(&mut code_it) {
                None => {
                    if let Some(p) = pattern_it.next() {
                        fail_match!("Part of the pattern was unmatched: {:?}", p);
                    }
                    return Ok(());
                }
                Some(SyntaxElement::Token(c)) => {
                    self.attempt_match_token(phase, &mut pattern_it, &c)?;
                }
                Some(SyntaxElement::Node(c)) => match pattern_it.next() {
                    Some(SyntaxElement::Node(p)) => {
                        self.attempt_match_node(phase, &p, &c)?;
                    }
                    Some(p) => fail_match!("Pattern wanted '{}', code has {}", p, c.text()),
                    None => fail_match!("Pattern reached end, code has {}", c.text()),
                },
            }
        }
    }

    fn attempt_match_token(
        &self,
        phase: &mut Phase<'_>,
        pattern: &mut Peekable<PatternIterator>,
        code: &syntax::SyntaxToken,
    ) -> Result<(), MatchFailed> {
        phase.record_ignored_comments(code);
        // Ignore whitespace and comments.
        if code.kind().is_trivia() {
            return Ok(());
        }
        if let Some(SyntaxElement::Token(p)) = pattern.peek() {
            // If the code has a comma and the pattern is about to close something, then accept the
            // comma without advancing the pattern. i.e. ignore trailing commas.
            if code.kind() == SyntaxKind::COMMA && is_closing_token(p.kind()) {
                return Ok(());
            }
            // Conversely, if the pattern has a comma and the code doesn't, skip that part of the
            // pattern and continue to match the code.
            if p.kind() == SyntaxKind::COMMA && is_closing_token(code.kind()) {
                pattern.next();
            }
        }
        // Consume an element from the pattern and make sure it matches.
        match pattern.next() {
            Some(SyntaxElement::Token(p)) => {
                if p.kind() != code.kind() || p.text() != code.text() {
                    fail_match!(
                        "Pattern wanted token '{}' ({:?}), but code had token '{}' ({:?})",
                        p.text(),
                        p.kind(),
                        code.text(),
                        code.kind()
                    )
                }
            }
            Some(SyntaxElement::Node(p)) => {
                // Not sure if this is actually reachable.
                fail_match!(
                    "Pattern wanted {:?}, but code had token '{}' ({:?})",
                    p,
                    code.text(),
                    code.kind()
                );
            }
            None => {
                fail_match!("Pattern exhausted, while code remains: `{}`", code.text());
            }
        }
        Ok(())
    }

    #[allow(clippy::only_used_in_recursion)]
    fn check_constraint(
        &self,
        constraint: &Constraint,
        code: &SyntaxNode,
    ) -> Result<(), MatchFailed> {
        match constraint {
            Constraint::Kind(kind) => {
                kind.matches(code)?;
            }
            Constraint::Not(sub) => {
                if self.check_constraint(sub, code).is_ok() {
                    fail_match!("Constraint {:?} failed for '{}'", constraint, code.text());
                }
            }
        }
        Ok(())
    }

    /// Paths are matched based on whether they refer to the same thing, even if they're written
    /// differently.
    fn attempt_match_path(
        &self,
        phase: &mut Phase<'_>,
        pattern: &SyntaxNode,
        code: &SyntaxNode,
    ) -> Result<(), MatchFailed> {
        if let Some(pattern_resolved) = self.rule.pattern.resolved_paths.get(pattern) {
            let pattern_path = ast::Path::cast(pattern.clone()).unwrap();
            let code_path = ast::Path::cast(code.clone()).unwrap();
            if let (Some(pattern_segment), Some(code_segment)) =
                (pattern_path.segment(), code_path.segment())
            {
                // Match everything within the segment except for the name-ref, which is handled
                // separately via comparing what the path resolves to below.
                self.attempt_match_opt(
                    phase,
                    pattern_segment.generic_arg_list(),
                    code_segment.generic_arg_list(),
                )?;
                self.attempt_match_opt(
                    phase,
                    pattern_segment.parenthesized_arg_list(),
                    code_segment.parenthesized_arg_list(),
                )?;
            }
            if matches!(phase, Phase::Second(_)) {
                let resolution = self
                    .sema
                    .resolve_path(&code_path)
                    .ok_or_else(|| match_error!("Failed to resolve path `{}`", code.text()))?;
                if pattern_resolved.resolution != resolution {
                    fail_match!("Pattern had path `{}` code had `{}`", pattern.text(), code.text());
                }
            }
        } else {
            return self.attempt_match_node_children(phase, pattern, code);
        }
        Ok(())
    }

    fn attempt_match_opt<T: AstNode>(
        &self,
        phase: &mut Phase<'_>,
        pattern: Option<T>,
        code: Option<T>,
    ) -> Result<(), MatchFailed> {
        match (pattern, code) {
            (Some(p), Some(c)) => self.attempt_match_node(phase, p.syntax(), c.syntax()),
            (None, None) => Ok(()),
            (Some(p), None) => fail_match!("Pattern `{}` had nothing to match", p.syntax().text()),
            (None, Some(c)) => {
                fail_match!("Nothing in pattern to match code `{}`", c.syntax().text())
            }
        }
    }

    /// We want to allow the records to match in any order, so we have special matching logic for
    /// them.
    fn attempt_match_record_field_list(
        &self,
        phase: &mut Phase<'_>,
        pattern: &SyntaxNode,
        code: &SyntaxNode,
    ) -> Result<(), MatchFailed> {
        // Build a map keyed by field name.
        let mut fields_by_name: FxHashMap<SmolStr, SyntaxNode> = FxHashMap::default();
        for child in code.children() {
            if let Some(record) = ast::RecordExprField::cast(child.clone()) {
                if let Some(name) = record.field_name() {
                    fields_by_name.insert(name.text().into(), child.clone());
                }
            }
        }
        for p in pattern.children_with_tokens() {
            if let SyntaxElement::Node(p) = p {
                if let Some(name_element) = p.first_child_or_token() {
                    if self.get_placeholder(&name_element).is_some() {
                        // If the pattern is using placeholders for field names then order
                        // independence doesn't make sense. Fall back to regular ordered
                        // matching.
                        return self.attempt_match_node_children(phase, pattern, code);
                    }
                    if let Some(ident) = only_ident(name_element) {
                        let code_record = fields_by_name.remove(ident.text()).ok_or_else(|| {
                            match_error!(
                                "Placeholder has record field '{}', but code doesn't",
                                ident
                            )
                        })?;
                        self.attempt_match_node(phase, &p, &code_record)?;
                    }
                }
            }
        }
        if let Some(unmatched_fields) = fields_by_name.keys().next() {
            fail_match!(
                "{} field(s) of a record literal failed to match, starting with {}",
                fields_by_name.len(),
                unmatched_fields
            );
        }
        Ok(())
    }

    /// Outside of token trees, a placeholder can only match a single AST node, whereas in a token
    /// tree it can match a sequence of tokens. Note, that this code will only be used when the
    /// pattern matches the macro invocation. For matches within the macro call, we'll already have
    /// expanded the macro.
    fn attempt_match_token_tree(
        &self,
        phase: &mut Phase<'_>,
        pattern: &SyntaxNode,
        code: &syntax::SyntaxNode,
    ) -> Result<(), MatchFailed> {
        let mut pattern = PatternIterator::new(pattern).peekable();
        let mut children = code.children_with_tokens();
        while let Some(child) = children.next() {
            if let Some(placeholder) = pattern.peek().and_then(|p| self.get_placeholder(p)) {
                pattern.next();
                let next_pattern_token = pattern
                    .peek()
                    .and_then(|p| match p {
                        SyntaxElement::Token(t) => Some(t.clone()),
                        SyntaxElement::Node(n) => n.first_token(),
                    })
                    .map(|p| p.text().to_owned());
                let first_matched_token = child.clone();
                let mut last_matched_token = child;
                // Read code tokens util we reach one equal to the next token from our pattern
                // or we reach the end of the token tree.
                for next in &mut children {
                    match &next {
                        SyntaxElement::Token(t) => {
                            if Some(t.to_string()) == next_pattern_token {
                                pattern.next();
                                break;
                            }
                        }
                        SyntaxElement::Node(n) => {
                            if let Some(first_token) = n.first_token() {
                                if Some(first_token.text()) == next_pattern_token.as_deref() {
                                    if let Some(SyntaxElement::Node(p)) = pattern.next() {
                                        // We have a subtree that starts with the next token in our pattern.
                                        self.attempt_match_token_tree(phase, &p, n)?;
                                        break;
                                    }
                                }
                            }
                        }
                    };
                    last_matched_token = next;
                }
                if let Phase::Second(match_out) = phase {
                    match_out.placeholder_values.insert(
                        placeholder.ident.clone(),
                        PlaceholderMatch::from_range(FileRange {
                            file_id: self
                                .sema
                                .original_range_opt(code)
                                .ok_or(MatchFailed {
                                    reason: Some("def site definition".to_owned()),
                                })?
                                .file_id,
                            range: first_matched_token
                                .text_range()
                                .cover(last_matched_token.text_range()),
                        }),
                    );
                }
                continue;
            }
            // Match literal (non-placeholder) tokens.
            match child {
                SyntaxElement::Token(token) => {
                    self.attempt_match_token(phase, &mut pattern, &token)?;
                }
                SyntaxElement::Node(node) => match pattern.next() {
                    Some(SyntaxElement::Node(p)) => {
                        self.attempt_match_token_tree(phase, &p, &node)?;
                    }
                    Some(SyntaxElement::Token(p)) => fail_match!(
                        "Pattern has token '{}', code has subtree '{}'",
                        p.text(),
                        node.text()
                    ),
                    None => fail_match!("Pattern has nothing, code has '{}'", node.text()),
                },
            }
        }
        if let Some(p) = pattern.next() {
            fail_match!("Reached end of token tree in code, but pattern still has {:?}", p);
        }
        Ok(())
    }

    fn attempt_match_ufcs_to_method_call(
        &self,
        phase: &mut Phase<'_>,
        pattern_ufcs: &UfcsCallInfo<'db>,
        code: &ast::MethodCallExpr,
    ) -> Result<(), MatchFailed> {
        use ast::HasArgList;
        let code_resolved_function = self
            .sema
            .resolve_method_call(code)
            .ok_or_else(|| match_error!("Failed to resolve method call"))?;
        if pattern_ufcs.function != code_resolved_function {
            fail_match!("Method call resolved to a different function");
        }
        // Check arguments.
        let mut pattern_args = pattern_ufcs
            .call_expr
            .arg_list()
            .ok_or_else(|| match_error!("Pattern function call has no args"))?
            .args();
        // If the function we're calling takes a self parameter, then we store additional
        // information on the placeholder match about autoderef and autoref. This allows us to use
        // the placeholder in a context where autoderef and autoref don't apply.
        if code_resolved_function.self_param(self.sema.db).is_some() {
            if let (Some(pattern_type), Some(expr)) =
                (&pattern_ufcs.qualifier_type, &code.receiver())
            {
                let deref_count = self.check_expr_type(pattern_type, expr)?;
                let pattern_receiver = pattern_args.next();
                self.attempt_match_opt(phase, pattern_receiver.clone(), code.receiver())?;
                if let Phase::Second(match_out) = phase {
                    if let Some(placeholder_value) = pattern_receiver
                        .and_then(|n| self.get_placeholder_for_node(n.syntax()))
                        .and_then(|placeholder| {
                            match_out.placeholder_values.get_mut(&placeholder.ident)
                        })
                    {
                        placeholder_value.autoderef_count = deref_count;
                        placeholder_value.autoref_kind = self
                            .sema
                            .resolve_method_call_as_callable(code)
                            .and_then(|callable| {
                                let (self_param, _) = callable.receiver_param(self.sema.db)?;
                                Some(self.sema.source(self_param)?.value.kind())
                            })
                            .unwrap_or(ast::SelfParamKind::Owned);
                    }
                }
            }
        } else {
            self.attempt_match_opt(phase, pattern_args.next(), code.receiver())?;
        }
        let mut code_args =
            code.arg_list().ok_or_else(|| match_error!("Code method call has no args"))?.args();
        loop {
            match (pattern_args.next(), code_args.next()) {
                (None, None) => return Ok(()),
                (p, c) => self.attempt_match_opt(phase, p, c)?,
            }
        }
    }

    fn attempt_match_ufcs_to_ufcs(
        &self,
        phase: &mut Phase<'_>,
        pattern_ufcs: &UfcsCallInfo<'db>,
        code: &ast::CallExpr,
    ) -> Result<(), MatchFailed> {
        use ast::HasArgList;
        // Check that the first argument is the expected type.
        if let (Some(pattern_type), Some(expr)) = (
            &pattern_ufcs.qualifier_type,
            &code.arg_list().and_then(|code_args| code_args.args().next()),
        ) {
            self.check_expr_type(pattern_type, expr)?;
        }
        self.attempt_match_node_children(phase, pattern_ufcs.call_expr.syntax(), code.syntax())
    }

    /// Verifies that `expr` matches `pattern_type`, possibly after dereferencing some number of
    /// times. Returns the number of times it needed to be dereferenced.
    fn check_expr_type(
        &self,
        pattern_type: &hir::Type<'db>,
        expr: &ast::Expr,
    ) -> Result<usize, MatchFailed> {
        use hir::HirDisplay;
        let code_type = self
            .sema
            .type_of_expr(expr)
            .ok_or_else(|| {
                match_error!("Failed to get receiver type for `{}`", expr.syntax().text())
            })?
            .original;
        let krate = self.sema.scope(expr.syntax()).map(|it| it.krate()).unwrap_or_else(|| {
            hir::Crate::from(*self.sema.db.all_crates().last().expect("no crate graph present"))
        });

        code_type
            .autoderef(self.sema.db)
            .enumerate()
            .find(|(_, deref_code_type)| pattern_type == deref_code_type)
            .map(|(count, _)| count)
            .ok_or_else(|| {
                let display_target = krate.to_display_target(self.sema.db);
                // Temporary needed to make the borrow checker happy.
                match_error!(
                    "Pattern type `{}` didn't match code type `{}`",
                    pattern_type.display(self.sema.db, display_target),
                    code_type.display(self.sema.db, display_target)
                )
            })
    }

    fn get_placeholder_for_node(&self, node: &SyntaxNode) -> Option<&Placeholder> {
        self.get_placeholder(&SyntaxElement::Node(node.clone()))
    }

    fn get_placeholder(&self, element: &SyntaxElement) -> Option<&Placeholder> {
        only_ident(element.clone()).and_then(|ident| self.rule.get_placeholder(&ident))
    }
}

impl Match {
    fn render_template_paths<'db>(
        &mut self,
        template: &ResolvedPattern<'db>,
        sema: &Semantics<'db, ide_db::RootDatabase>,
    ) -> Result<(), MatchFailed> {
        let module = sema
            .scope(&self.matched_node)
            .ok_or_else(|| match_error!("Matched node isn't in a module"))?
            .module();
        for (path, resolved_path) in &template.resolved_paths {
            if let hir::PathResolution::Def(module_def) = resolved_path.resolution {
                let cfg = ImportPathConfig {
                    prefer_no_std: false,
                    prefer_prelude: true,
                    prefer_absolute: false,
                    allow_unstable: true,
                };
                let mod_path = module.find_path(sema.db, module_def, cfg).ok_or_else(|| {
                    match_error!("Failed to render template path `{}` at match location")
                })?;
                self.rendered_template_paths.insert(path.clone(), mod_path);
            }
        }
        Ok(())
    }
}

impl Phase<'_> {
    fn next_non_trivial(&mut self, code_it: &mut SyntaxElementChildren) -> Option<SyntaxElement> {
        loop {
            let c = code_it.next();
            if let Some(SyntaxElement::Token(t)) = &c {
                self.record_ignored_comments(t);
                if t.kind().is_trivia() {
                    continue;
                }
            }
            return c;
        }
    }

    fn record_ignored_comments(&mut self, token: &SyntaxToken) {
        if token.kind() == SyntaxKind::COMMENT {
            if let Phase::Second(match_out) = self {
                if let Some(comment) = ast::Comment::cast(token.clone()) {
                    match_out.ignored_comments.push(comment);
                }
            }
        }
    }
}

fn is_closing_token(kind: SyntaxKind) -> bool {
    kind == SyntaxKind::R_PAREN || kind == SyntaxKind::R_CURLY || kind == SyntaxKind::R_BRACK
}

pub(crate) fn record_match_fails_reasons_scope<F, T>(debug_active: bool, f: F) -> T
where
    F: Fn() -> T,
{
    RECORDING_MATCH_FAIL_REASONS.with(|c| c.set(debug_active));
    let res = f();
    RECORDING_MATCH_FAIL_REASONS.with(|c| c.set(false));
    res
}

// For performance reasons, we don't want to record the reason why every match fails, only the bit
// of code that the user indicated they thought would match. We use a thread local to indicate when
// we are trying to match that bit of code. This saves us having to pass a boolean into all the bits
// of code that can make the decision to not match.
thread_local! {
    pub static RECORDING_MATCH_FAIL_REASONS: Cell<bool> = const { Cell::new(false) };
}

fn recording_match_fail_reasons() -> bool {
    RECORDING_MATCH_FAIL_REASONS.with(|c| c.get())
}

impl PlaceholderMatch {
    fn from_range(range: FileRange) -> Self {
        Self {
            range,
            inner_matches: SsrMatches::default(),
            autoderef_count: 0,
            autoref_kind: ast::SelfParamKind::Owned,
        }
    }
}

impl NodeKind {
    fn matches(&self, node: &SyntaxNode) -> Result<(), MatchFailed> {
        let ok = match self {
            Self::Literal => {
                cov_mark::hit!(literal_constraint);
                ast::Literal::can_cast(node.kind())
            }
        };
        if !ok {
            fail_match!("Code '{}' isn't of kind {:?}", node.text(), self);
        }
        Ok(())
    }
}

// If `node` contains nothing but an ident then return it, otherwise return None.
fn only_ident(element: SyntaxElement) -> Option<SyntaxToken> {
    match element {
        SyntaxElement::Token(t) => {
            if t.kind() == SyntaxKind::IDENT {
                return Some(t);
            }
        }
        SyntaxElement::Node(n) => {
            let mut children = n.children_with_tokens();
            if let (Some(only_child), None) = (children.next(), children.next()) {
                return only_ident(only_child);
            }
        }
    }
    None
}

struct PatternIterator {
    iter: SyntaxElementChildren,
}

impl Iterator for PatternIterator {
    type Item = SyntaxElement;

    fn next(&mut self) -> Option<SyntaxElement> {
        self.iter.find(|element| !element.kind().is_trivia())
    }
}

impl PatternIterator {
    fn new(parent: &SyntaxNode) -> Self {
        Self { iter: parent.children_with_tokens() }
    }
}

#[cfg(test)]
mod tests {
    use crate::{MatchFinder, SsrRule};

    #[test]
    fn parse_match_replace() {
        let rule: SsrRule = "foo($x) ==>> bar($x)".parse().unwrap();
        let input = "fn foo() {} fn bar() {} fn main() { foo(1+2); }";

        let (db, position, selections) = crate::tests::single_file(input);
        let position = ide_db::FilePosition {
            file_id: position.file_id.file_id(&db),
            offset: position.offset,
        };
        let mut match_finder = MatchFinder::in_context(
            &db,
            position,
            selections
                .into_iter()
                .map(|frange| ide_db::FileRange {
                    file_id: frange.file_id.file_id(&db),
                    range: frange.range,
                })
                .collect(),
        )
        .unwrap();
        match_finder.add_rule(rule).unwrap();
        let matches = match_finder.matches();
        assert_eq!(matches.matches.len(), 1);
        assert_eq!(matches.matches[0].matched_node.text(), "foo(1+2)");
        assert_eq!(matches.matches[0].placeholder_values.len(), 1);

        let edits = match_finder.edits();
        assert_eq!(edits.len(), 1);
        let edit = &edits[&position.file_id];
        let mut after = input.to_owned();
        edit.apply(&mut after);
        assert_eq!(after, "fn foo() {} fn bar() {} fn main() { bar(1+2); }");
    }
}
