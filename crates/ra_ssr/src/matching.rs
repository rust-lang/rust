//! This module is responsible for matching a search pattern against a node in the AST. In the
//! process of matching, placeholder values are recorded.

use crate::{
    parsing::{Constraint, NodeKind, Placeholder, SsrTemplate},
    SsrMatches, SsrPattern, SsrRule,
};
use hir::Semantics;
use ra_db::FileRange;
use ra_syntax::ast::{AstNode, AstToken};
use ra_syntax::{ast, SyntaxElement, SyntaxElementChildren, SyntaxKind, SyntaxNode, SyntaxToken};
use rustc_hash::FxHashMap;
use std::{cell::Cell, iter::Peekable};
use test_utils::mark;

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
    // A copy of the template for the rule that produced this match. We store this on the match for
    // if/when we do replacement.
    pub(crate) template: SsrTemplate,
}

/// Represents a `$var` in an SSR query.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct Var(pub String);

/// Information about a placeholder bound in a match.
#[derive(Debug)]
pub(crate) struct PlaceholderMatch {
    /// The node that the placeholder matched to. If set, then we'll search for further matches
    /// within this node. It isn't set when we match tokens within a macro call's token tree.
    pub(crate) node: Option<SyntaxNode>,
    pub(crate) range: FileRange,
    /// More matches, found within `node`.
    pub(crate) inner_matches: SsrMatches,
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
pub(crate) fn get_match(
    debug_active: bool,
    rule: &SsrRule,
    code: &SyntaxNode,
    restrict_range: &Option<FileRange>,
    sema: &Semantics<ra_ide_db::RootDatabase>,
) -> Result<Match, MatchFailed> {
    record_match_fails_reasons_scope(debug_active, || {
        Matcher::try_match(rule, code, restrict_range, sema)
    })
}

/// Checks if our search pattern matches a particular node of the AST.
struct Matcher<'db, 'sema> {
    sema: &'sema Semantics<'db, ra_ide_db::RootDatabase>,
    /// If any placeholders come from anywhere outside of this range, then the match will be
    /// rejected.
    restrict_range: Option<FileRange>,
    rule: &'sema SsrRule,
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
        rule: &'sema SsrRule,
        code: &SyntaxNode,
        restrict_range: &Option<FileRange>,
        sema: &'sema Semantics<'db, ra_ide_db::RootDatabase>,
    ) -> Result<Match, MatchFailed> {
        let match_state = Matcher { sema, restrict_range: restrict_range.clone(), rule };
        let pattern_tree = rule.pattern.tree_for_kind(code.kind())?;
        // First pass at matching, where we check that node types and idents match.
        match_state.attempt_match_node(&mut Phase::First, &pattern_tree, code)?;
        match_state.validate_range(&sema.original_range(code))?;
        let mut the_match = Match {
            range: sema.original_range(code),
            matched_node: code.clone(),
            placeholder_values: FxHashMap::default(),
            ignored_comments: Vec::new(),
            template: rule.template.clone(),
        };
        // Second matching pass, where we record placeholder matches, ignored comments and maybe do
        // any other more expensive checks that we didn't want to do on the first pass.
        match_state.attempt_match_node(&mut Phase::Second(&mut the_match), &pattern_tree, code)?;
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
        phase: &mut Phase,
        pattern: &SyntaxNode,
        code: &SyntaxNode,
    ) -> Result<(), MatchFailed> {
        // Handle placeholders.
        if let Some(placeholder) = self.get_placeholder(&SyntaxElement::Node(pattern.clone())) {
            for constraint in &placeholder.constraints {
                self.check_constraint(constraint, code)?;
            }
            if let Phase::Second(matches_out) = phase {
                let original_range = self.sema.original_range(code);
                // We validated the range for the node when we started the match, so the placeholder
                // probably can't fail range validation, but just to be safe...
                self.validate_range(&original_range)?;
                matches_out.placeholder_values.insert(
                    Var(placeholder.ident.to_string()),
                    PlaceholderMatch::new(code, original_range),
                );
            }
            return Ok(());
        }
        // Non-placeholders.
        if pattern.kind() != code.kind() {
            fail_match!(
                "Pattern had a `{}` ({:?}), code had `{}` ({:?})",
                pattern.text(),
                pattern.kind(),
                code.text(),
                code.kind()
            );
        }
        // Some kinds of nodes have special handling. For everything else, we fall back to default
        // matching.
        match code.kind() {
            SyntaxKind::RECORD_FIELD_LIST => {
                self.attempt_match_record_field_list(phase, pattern, code)
            }
            SyntaxKind::TOKEN_TREE => self.attempt_match_token_tree(phase, pattern, code),
            _ => self.attempt_match_node_children(phase, pattern, code),
        }
    }

    fn attempt_match_node_children(
        &self,
        phase: &mut Phase,
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
        phase: &mut Phase,
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
        phase: &mut Phase,
        pattern: &mut Peekable<PatternIterator>,
        code: &ra_syntax::SyntaxToken,
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
                if self.check_constraint(&*sub, code).is_ok() {
                    fail_match!("Constraint {:?} failed for '{}'", constraint, code.text());
                }
            }
        }
        Ok(())
    }

    /// We want to allow the records to match in any order, so we have special matching logic for
    /// them.
    fn attempt_match_record_field_list(
        &self,
        phase: &mut Phase,
        pattern: &SyntaxNode,
        code: &SyntaxNode,
    ) -> Result<(), MatchFailed> {
        // Build a map keyed by field name.
        let mut fields_by_name = FxHashMap::default();
        for child in code.children() {
            if let Some(record) = ast::RecordField::cast(child.clone()) {
                if let Some(name) = record.field_name() {
                    fields_by_name.insert(name.text().clone(), child.clone());
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
        phase: &mut Phase,
        pattern: &SyntaxNode,
        code: &ra_syntax::SyntaxNode,
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
                    .map(|p| p.text().to_string());
                let first_matched_token = child.clone();
                let mut last_matched_token = child;
                // Read code tokens util we reach one equal to the next token from our pattern
                // or we reach the end of the token tree.
                while let Some(next) = children.next() {
                    match &next {
                        SyntaxElement::Token(t) => {
                            if Some(t.to_string()) == next_pattern_token {
                                pattern.next();
                                break;
                            }
                        }
                        SyntaxElement::Node(n) => {
                            if let Some(first_token) = n.first_token() {
                                if Some(first_token.to_string()) == next_pattern_token {
                                    if let Some(SyntaxElement::Node(p)) = pattern.next() {
                                        // We have a subtree that starts with the next token in our pattern.
                                        self.attempt_match_token_tree(phase, &p, &n)?;
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
                        Var(placeholder.ident.to_string()),
                        PlaceholderMatch::from_range(FileRange {
                            file_id: self.sema.original_range(code).file_id,
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

    fn get_placeholder(&self, element: &SyntaxElement) -> Option<&Placeholder> {
        only_ident(element.clone())
            .and_then(|ident| self.rule.pattern.placeholders_by_stand_in.get(ident.text()))
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
    pub static RECORDING_MATCH_FAIL_REASONS: Cell<bool> = Cell::new(false);
}

fn recording_match_fail_reasons() -> bool {
    RECORDING_MATCH_FAIL_REASONS.with(|c| c.get())
}

impl PlaceholderMatch {
    fn new(node: &SyntaxNode, range: FileRange) -> Self {
        Self { node: Some(node.clone()), range, inner_matches: SsrMatches::default() }
    }

    fn from_range(range: FileRange) -> Self {
        Self { node: None, range, inner_matches: SsrMatches::default() }
    }
}

impl SsrPattern {
    pub(crate) fn tree_for_kind(&self, kind: SyntaxKind) -> Result<&SyntaxNode, MatchFailed> {
        let (tree, kind_name) = if ast::Expr::can_cast(kind) {
            (&self.expr, "expression")
        } else if ast::TypeRef::can_cast(kind) {
            (&self.type_ref, "type reference")
        } else if ast::ModuleItem::can_cast(kind) {
            (&self.item, "item")
        } else if ast::Path::can_cast(kind) {
            (&self.path, "path")
        } else if ast::Pat::can_cast(kind) {
            (&self.pattern, "pattern")
        } else {
            fail_match!("Matching nodes of kind {:?} is not supported", kind);
        };
        match tree {
            Some(tree) => Ok(tree),
            None => fail_match!("Pattern cannot be parsed as a {}", kind_name),
        }
    }
}

impl NodeKind {
    fn matches(&self, node: &SyntaxNode) -> Result<(), MatchFailed> {
        let ok = match self {
            Self::Literal => {
                mark::hit!(literal_constraint);
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
        while let Some(element) = self.iter.next() {
            if !element.kind().is_trivia() {
                return Some(element);
            }
        }
        None
    }
}

impl PatternIterator {
    fn new(parent: &SyntaxNode) -> Self {
        Self { iter: parent.children_with_tokens() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MatchFinder, SsrRule};

    #[test]
    fn parse_match_replace() {
        let rule: SsrRule = "foo($x) ==>> bar($x)".parse().unwrap();
        let input = "fn foo() {} fn main() { foo(1+2); }";

        use ra_db::fixture::WithFixture;
        let (db, file_id) = ra_ide_db::RootDatabase::with_single_file(input);
        let mut match_finder = MatchFinder::new(&db);
        match_finder.add_rule(rule);
        let matches = match_finder.find_matches_in_file(file_id);
        assert_eq!(matches.matches.len(), 1);
        assert_eq!(matches.matches[0].matched_node.text(), "foo(1+2)");
        assert_eq!(matches.matches[0].placeholder_values.len(), 1);
        assert_eq!(
            matches.matches[0].placeholder_values[&Var("x".to_string())]
                .node
                .as_ref()
                .unwrap()
                .text(),
            "1+2"
        );

        let edit = crate::replacing::matches_to_edit(&matches, input);
        let mut after = input.to_string();
        edit.apply(&mut after);
        assert_eq!(after, "fn foo() {} fn main() { bar(1+2); }");
    }
}
