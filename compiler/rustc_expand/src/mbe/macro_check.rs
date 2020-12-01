//! Checks that meta-variables in macro definition are correctly declared and used.
//!
//! # What is checked
//!
//! ## Meta-variables must not be bound twice
//!
//! ```
//! macro_rules! foo { ($x:tt $x:tt) => { $x }; }
//! ```
//!
//! This check is sound (no false-negative) and complete (no false-positive).
//!
//! ## Meta-variables must not be free
//!
//! ```
//! macro_rules! foo { () => { $x }; }
//! ```
//!
//! This check is also done at macro instantiation but only if the branch is taken.
//!
//! ## Meta-variables must repeat at least as many times as their binder
//!
//! ```
//! macro_rules! foo { ($($x:tt)*) => { $x }; }
//! ```
//!
//! This check is also done at macro instantiation but only if the branch is taken.
//!
//! ## Meta-variables must repeat with the same Kleene operators as their binder
//!
//! ```
//! macro_rules! foo { ($($x:tt)+) => { $($x)* }; }
//! ```
//!
//! This check is not done at macro instantiation.
//!
//! # Disclaimer
//!
//! In the presence of nested macros (a macro defined in a macro), those checks may have false
//! positives and false negatives. We try to detect those cases by recognizing potential macro
//! definitions in RHSes, but nested macros may be hidden through the use of particular values of
//! meta-variables.
//!
//! ## Examples of false positive
//!
//! False positives can come from cases where we don't recognize a nested macro, because it depends
//! on particular values of meta-variables. In the following example, we think both instances of
//! `$x` are free, which is a correct statement if `$name` is anything but `macro_rules`. But when
//! `$name` is `macro_rules`, like in the instantiation below, then `$x:tt` is actually a binder of
//! the nested macro and `$x` is bound to it.
//!
//! ```
//! macro_rules! foo { ($name:ident) => { $name! bar { ($x:tt) => { $x }; } }; }
//! foo!(macro_rules);
//! ```
//!
//! False positives can also come from cases where we think there is a nested macro while there
//! isn't. In the following example, we think `$x` is free, which is incorrect because `bar` is not
//! a nested macro since it is not evaluated as code by `stringify!`.
//!
//! ```
//! macro_rules! foo { () => { stringify!(macro_rules! bar { () => { $x }; }) }; }
//! ```
//!
//! ## Examples of false negative
//!
//! False negatives can come from cases where we don't recognize a meta-variable, because it depends
//! on particular values of meta-variables. In the following examples, we don't see that if `$d` is
//! instantiated with `$` then `$d z` becomes `$z` in the nested macro definition and is thus a free
//! meta-variable. Note however, that if `foo` is instantiated, then we would check the definition
//! of `bar` and would see the issue.
//!
//! ```
//! macro_rules! foo { ($d:tt) => { macro_rules! bar { ($y:tt) => { $d z }; } }; }
//! ```
//!
//! # How it is checked
//!
//! There are 3 main functions: `check_binders`, `check_occurrences`, and `check_nested_macro`. They
//! all need some kind of environment.
//!
//! ## Environments
//!
//! Environments are used to pass information.
//!
//! ### From LHS to RHS
//!
//! When checking a LHS with `check_binders`, we produce (and use) an environment for binders,
//! namely `Binders`. This is a mapping from binder name to information about that binder: the span
//! of the binder for error messages and the stack of Kleene operators under which it was bound in
//! the LHS.
//!
//! This environment is used by both the LHS and RHS. The LHS uses it to detect duplicate binders.
//! The RHS uses it to detect the other errors.
//!
//! ### From outer macro to inner macro
//!
//! When checking the RHS of an outer macro and we detect a nested macro definition, we push the
//! current state, namely `MacroState`, to an environment of nested macro definitions. Each state
//! stores the LHS binders when entering the macro definition as well as the stack of Kleene
//! operators under which the inner macro is defined in the RHS.
//!
//! This environment is a stack representing the nesting of macro definitions. As such, the stack of
//! Kleene operators under which a meta-variable is repeating is the concatenation of the stacks
//! stored when entering a macro definition starting from the state in which the meta-variable is
//! bound.
use crate::mbe::{KleeneToken, TokenTree};

use rustc_ast::token::{DelimToken, Token, TokenKind};
use rustc_ast::{NodeId, DUMMY_NODE_ID};
use rustc_data_structures::fx::FxHashMap;
use rustc_session::lint::builtin::META_VARIABLE_MISUSE;
use rustc_session::parse::ParseSess;
use rustc_span::symbol::kw;
use rustc_span::{symbol::MacroRulesNormalizedIdent, MultiSpan, Span};

use smallvec::SmallVec;

/// Stack represented as linked list.
///
/// Those are used for environments because they grow incrementally and are not mutable.
enum Stack<'a, T> {
    /// Empty stack.
    Empty,
    /// A non-empty stack.
    Push {
        /// The top element.
        top: T,
        /// The previous elements.
        prev: &'a Stack<'a, T>,
    },
}

impl<'a, T> Stack<'a, T> {
    /// Returns whether a stack is empty.
    fn is_empty(&self) -> bool {
        matches!(*self, Stack::Empty)
    }

    /// Returns a new stack with an element of top.
    fn push(&'a self, top: T) -> Stack<'a, T> {
        Stack::Push { top, prev: self }
    }
}

impl<'a, T> Iterator for &'a Stack<'a, T> {
    type Item = &'a T;

    // Iterates from top to bottom of the stack.
    fn next(&mut self) -> Option<&'a T> {
        match *self {
            Stack::Empty => None,
            Stack::Push { ref top, ref prev } => {
                *self = prev;
                Some(top)
            }
        }
    }
}

impl From<&Stack<'_, KleeneToken>> for SmallVec<[KleeneToken; 1]> {
    fn from(ops: &Stack<'_, KleeneToken>) -> SmallVec<[KleeneToken; 1]> {
        let mut ops: SmallVec<[KleeneToken; 1]> = ops.cloned().collect();
        // The stack is innermost on top. We want outermost first.
        ops.reverse();
        ops
    }
}

/// Information attached to a meta-variable binder in LHS.
struct BinderInfo {
    /// The span of the meta-variable in LHS.
    span: Span,
    /// The stack of Kleene operators (outermost first).
    ops: SmallVec<[KleeneToken; 1]>,
}

/// An environment of meta-variables to their binder information.
type Binders = FxHashMap<MacroRulesNormalizedIdent, BinderInfo>;

/// The state at which we entered a macro definition in the RHS of another macro definition.
struct MacroState<'a> {
    /// The binders of the branch where we entered the macro definition.
    binders: &'a Binders,
    /// The stack of Kleene operators (outermost first) where we entered the macro definition.
    ops: SmallVec<[KleeneToken; 1]>,
}

/// Checks that meta-variables are used correctly in a macro definition.
///
/// Arguments:
/// - `sess` is used to emit diagnostics and lints
/// - `node_id` is used to emit lints
/// - `span` is used when no spans are available
/// - `lhses` and `rhses` should have the same length and represent the macro definition
pub(super) fn check_meta_variables(
    sess: &ParseSess,
    node_id: NodeId,
    span: Span,
    lhses: &[TokenTree],
    rhses: &[TokenTree],
) -> bool {
    if lhses.len() != rhses.len() {
        sess.span_diagnostic.span_bug(span, "length mismatch between LHSes and RHSes")
    }
    let mut valid = true;
    for (lhs, rhs) in lhses.iter().zip(rhses.iter()) {
        let mut binders = Binders::default();
        check_binders(sess, node_id, lhs, &Stack::Empty, &mut binders, &Stack::Empty, &mut valid);
        check_occurrences(sess, node_id, rhs, &Stack::Empty, &binders, &Stack::Empty, &mut valid);
    }
    valid
}

/// Checks `lhs` as part of the LHS of a macro definition, extends `binders` with new binders, and
/// sets `valid` to false in case of errors.
///
/// Arguments:
/// - `sess` is used to emit diagnostics and lints
/// - `node_id` is used to emit lints
/// - `lhs` is checked as part of a LHS
/// - `macros` is the stack of possible outer macros
/// - `binders` contains the binders of the LHS
/// - `ops` is the stack of Kleene operators from the LHS
/// - `valid` is set in case of errors
fn check_binders(
    sess: &ParseSess,
    node_id: NodeId,
    lhs: &TokenTree,
    macros: &Stack<'_, MacroState<'_>>,
    binders: &mut Binders,
    ops: &Stack<'_, KleeneToken>,
    valid: &mut bool,
) {
    match *lhs {
        TokenTree::Token(..) => {}
        // This can only happen when checking a nested macro because this LHS is then in the RHS of
        // the outer macro. See ui/macros/macro-of-higher-order.rs where $y:$fragment in the
        // LHS of the nested macro (and RHS of the outer macro) is parsed as MetaVar(y) Colon
        // MetaVar(fragment) and not as MetaVarDecl(y, fragment).
        TokenTree::MetaVar(span, name) => {
            if macros.is_empty() {
                sess.span_diagnostic.span_bug(span, "unexpected MetaVar in lhs");
            }
            let name = MacroRulesNormalizedIdent::new(name);
            // There are 3 possibilities:
            if let Some(prev_info) = binders.get(&name) {
                // 1. The meta-variable is already bound in the current LHS: This is an error.
                let mut span = MultiSpan::from_span(span);
                span.push_span_label(prev_info.span, "previous declaration".into());
                buffer_lint(sess, span, node_id, "duplicate matcher binding");
            } else if get_binder_info(macros, binders, name).is_none() {
                // 2. The meta-variable is free: This is a binder.
                binders.insert(name, BinderInfo { span, ops: ops.into() });
            } else {
                // 3. The meta-variable is bound: This is an occurrence.
                check_occurrences(sess, node_id, lhs, macros, binders, ops, valid);
            }
        }
        // Similarly, this can only happen when checking a toplevel macro.
        TokenTree::MetaVarDecl(span, name, _kind) => {
            if !macros.is_empty() {
                sess.span_diagnostic.span_bug(span, "unexpected MetaVarDecl in nested lhs");
            }
            let name = MacroRulesNormalizedIdent::new(name);
            if let Some(prev_info) = get_binder_info(macros, binders, name) {
                // Duplicate binders at the top-level macro definition are errors. The lint is only
                // for nested macro definitions.
                sess.span_diagnostic
                    .struct_span_err(span, "duplicate matcher binding")
                    .span_label(span, "duplicate binding")
                    .span_label(prev_info.span, "previous binding")
                    .emit();
                *valid = false;
            } else {
                binders.insert(name, BinderInfo { span, ops: ops.into() });
            }
        }
        TokenTree::Delimited(_, ref del) => {
            for tt in &del.tts {
                check_binders(sess, node_id, tt, macros, binders, ops, valid);
            }
        }
        TokenTree::Sequence(_, ref seq) => {
            let ops = ops.push(seq.kleene);
            for tt in &seq.tts {
                check_binders(sess, node_id, tt, macros, binders, &ops, valid);
            }
        }
    }
}

/// Returns the binder information of a meta-variable.
///
/// Arguments:
/// - `macros` is the stack of possible outer macros
/// - `binders` contains the current binders
/// - `name` is the name of the meta-variable we are looking for
fn get_binder_info<'a>(
    mut macros: &'a Stack<'a, MacroState<'a>>,
    binders: &'a Binders,
    name: MacroRulesNormalizedIdent,
) -> Option<&'a BinderInfo> {
    binders.get(&name).or_else(|| macros.find_map(|state| state.binders.get(&name)))
}

/// Checks `rhs` as part of the RHS of a macro definition and sets `valid` to false in case of
/// errors.
///
/// Arguments:
/// - `sess` is used to emit diagnostics and lints
/// - `node_id` is used to emit lints
/// - `rhs` is checked as part of a RHS
/// - `macros` is the stack of possible outer macros
/// - `binders` contains the binders of the associated LHS
/// - `ops` is the stack of Kleene operators from the RHS
/// - `valid` is set in case of errors
fn check_occurrences(
    sess: &ParseSess,
    node_id: NodeId,
    rhs: &TokenTree,
    macros: &Stack<'_, MacroState<'_>>,
    binders: &Binders,
    ops: &Stack<'_, KleeneToken>,
    valid: &mut bool,
) {
    match *rhs {
        TokenTree::Token(..) => {}
        TokenTree::MetaVarDecl(span, _name, _kind) => {
            sess.span_diagnostic.span_bug(span, "unexpected MetaVarDecl in rhs")
        }
        TokenTree::MetaVar(span, name) => {
            let name = MacroRulesNormalizedIdent::new(name);
            check_ops_is_prefix(sess, node_id, macros, binders, ops, span, name);
        }
        TokenTree::Delimited(_, ref del) => {
            check_nested_occurrences(sess, node_id, &del.tts, macros, binders, ops, valid);
        }
        TokenTree::Sequence(_, ref seq) => {
            let ops = ops.push(seq.kleene);
            check_nested_occurrences(sess, node_id, &seq.tts, macros, binders, &ops, valid);
        }
    }
}

/// Represents the processed prefix of a nested macro.
#[derive(Clone, Copy, PartialEq, Eq)]
enum NestedMacroState {
    /// Nothing that matches a nested macro definition was processed yet.
    Empty,
    /// The token `macro_rules` was processed.
    MacroRules,
    /// The tokens `macro_rules!` were processed.
    MacroRulesNot,
    /// The tokens `macro_rules!` followed by a name were processed. The name may be either directly
    /// an identifier or a meta-variable (that hopefully would be instantiated by an identifier).
    MacroRulesNotName,
    /// The keyword `macro` was processed.
    Macro,
    /// The keyword `macro` followed by a name was processed.
    MacroName,
    /// The keyword `macro` followed by a name and a token delimited by parentheses was processed.
    MacroNameParen,
}

/// Checks `tts` as part of the RHS of a macro definition, tries to recognize nested macro
/// definitions, and sets `valid` to false in case of errors.
///
/// Arguments:
/// - `sess` is used to emit diagnostics and lints
/// - `node_id` is used to emit lints
/// - `tts` is checked as part of a RHS and may contain macro definitions
/// - `macros` is the stack of possible outer macros
/// - `binders` contains the binders of the associated LHS
/// - `ops` is the stack of Kleene operators from the RHS
/// - `valid` is set in case of errors
fn check_nested_occurrences(
    sess: &ParseSess,
    node_id: NodeId,
    tts: &[TokenTree],
    macros: &Stack<'_, MacroState<'_>>,
    binders: &Binders,
    ops: &Stack<'_, KleeneToken>,
    valid: &mut bool,
) {
    let mut state = NestedMacroState::Empty;
    let nested_macros = macros.push(MacroState { binders, ops: ops.into() });
    let mut nested_binders = Binders::default();
    for tt in tts {
        match (state, tt) {
            (
                NestedMacroState::Empty,
                &TokenTree::Token(Token { kind: TokenKind::Ident(name, false), .. }),
            ) => {
                if name == kw::MacroRules {
                    state = NestedMacroState::MacroRules;
                } else if name == kw::Macro {
                    state = NestedMacroState::Macro;
                }
            }
            (
                NestedMacroState::MacroRules,
                &TokenTree::Token(Token { kind: TokenKind::Not, .. }),
            ) => {
                state = NestedMacroState::MacroRulesNot;
            }
            (
                NestedMacroState::MacroRulesNot,
                &TokenTree::Token(Token { kind: TokenKind::Ident(..), .. }),
            ) => {
                state = NestedMacroState::MacroRulesNotName;
            }
            (NestedMacroState::MacroRulesNot, &TokenTree::MetaVar(..)) => {
                state = NestedMacroState::MacroRulesNotName;
                // We check that the meta-variable is correctly used.
                check_occurrences(sess, node_id, tt, macros, binders, ops, valid);
            }
            (NestedMacroState::MacroRulesNotName, &TokenTree::Delimited(_, ref del))
            | (NestedMacroState::MacroName, &TokenTree::Delimited(_, ref del))
                if del.delim == DelimToken::Brace =>
            {
                let macro_rules = state == NestedMacroState::MacroRulesNotName;
                state = NestedMacroState::Empty;
                let rest =
                    check_nested_macro(sess, node_id, macro_rules, &del.tts, &nested_macros, valid);
                // If we did not check the whole macro definition, then check the rest as if outside
                // the macro definition.
                check_nested_occurrences(
                    sess,
                    node_id,
                    &del.tts[rest..],
                    macros,
                    binders,
                    ops,
                    valid,
                );
            }
            (
                NestedMacroState::Macro,
                &TokenTree::Token(Token { kind: TokenKind::Ident(..), .. }),
            ) => {
                state = NestedMacroState::MacroName;
            }
            (NestedMacroState::Macro, &TokenTree::MetaVar(..)) => {
                state = NestedMacroState::MacroName;
                // We check that the meta-variable is correctly used.
                check_occurrences(sess, node_id, tt, macros, binders, ops, valid);
            }
            (NestedMacroState::MacroName, &TokenTree::Delimited(_, ref del))
                if del.delim == DelimToken::Paren =>
            {
                state = NestedMacroState::MacroNameParen;
                nested_binders = Binders::default();
                check_binders(
                    sess,
                    node_id,
                    tt,
                    &nested_macros,
                    &mut nested_binders,
                    &Stack::Empty,
                    valid,
                );
            }
            (NestedMacroState::MacroNameParen, &TokenTree::Delimited(_, ref del))
                if del.delim == DelimToken::Brace =>
            {
                state = NestedMacroState::Empty;
                check_occurrences(
                    sess,
                    node_id,
                    tt,
                    &nested_macros,
                    &nested_binders,
                    &Stack::Empty,
                    valid,
                );
            }
            (_, ref tt) => {
                state = NestedMacroState::Empty;
                check_occurrences(sess, node_id, tt, macros, binders, ops, valid);
            }
        }
    }
}

/// Checks the body of nested macro, returns where the check stopped, and sets `valid` to false in
/// case of errors.
///
/// The token trees are checked as long as they look like a list of (LHS) => {RHS} token trees. This
/// check is a best-effort to detect a macro definition. It returns the position in `tts` where we
/// stopped checking because we detected we were not in a macro definition anymore.
///
/// Arguments:
/// - `sess` is used to emit diagnostics and lints
/// - `node_id` is used to emit lints
/// - `macro_rules` specifies whether the macro is `macro_rules`
/// - `tts` is checked as a list of (LHS) => {RHS}
/// - `macros` is the stack of outer macros
/// - `valid` is set in case of errors
fn check_nested_macro(
    sess: &ParseSess,
    node_id: NodeId,
    macro_rules: bool,
    tts: &[TokenTree],
    macros: &Stack<'_, MacroState<'_>>,
    valid: &mut bool,
) -> usize {
    let n = tts.len();
    let mut i = 0;
    let separator = if macro_rules { TokenKind::Semi } else { TokenKind::Comma };
    loop {
        // We expect 3 token trees: `(LHS) => {RHS}`. The separator is checked after.
        if i + 2 >= n
            || !tts[i].is_delimited()
            || !tts[i + 1].is_token(&TokenKind::FatArrow)
            || !tts[i + 2].is_delimited()
        {
            break;
        }
        let lhs = &tts[i];
        let rhs = &tts[i + 2];
        let mut binders = Binders::default();
        check_binders(sess, node_id, lhs, macros, &mut binders, &Stack::Empty, valid);
        check_occurrences(sess, node_id, rhs, macros, &binders, &Stack::Empty, valid);
        // Since the last semicolon is optional for `macro_rules` macros and decl_macro are not terminated,
        // we increment our checked position by how many token trees we already checked (the 3
        // above) before checking for the separator.
        i += 3;
        if i == n || !tts[i].is_token(&separator) {
            break;
        }
        // We increment our checked position for the semicolon.
        i += 1;
    }
    i
}

/// Checks that a meta-variable occurrence is valid.
///
/// Arguments:
/// - `sess` is used to emit diagnostics and lints
/// - `node_id` is used to emit lints
/// - `macros` is the stack of possible outer macros
/// - `binders` contains the binders of the associated LHS
/// - `ops` is the stack of Kleene operators from the RHS
/// - `span` is the span of the meta-variable to check
/// - `name` is the name of the meta-variable to check
fn check_ops_is_prefix(
    sess: &ParseSess,
    node_id: NodeId,
    macros: &Stack<'_, MacroState<'_>>,
    binders: &Binders,
    ops: &Stack<'_, KleeneToken>,
    span: Span,
    name: MacroRulesNormalizedIdent,
) {
    let macros = macros.push(MacroState { binders, ops: ops.into() });
    // Accumulates the stacks the operators of each state until (and including when) the
    // meta-variable is found. The innermost stack is first.
    let mut acc: SmallVec<[&SmallVec<[KleeneToken; 1]>; 1]> = SmallVec::new();
    for state in &macros {
        acc.push(&state.ops);
        if let Some(binder) = state.binders.get(&name) {
            // This variable concatenates the stack of operators from the RHS of the LHS where the
            // meta-variable was defined to where it is used (in possibly nested macros). The
            // outermost operator is first.
            let mut occurrence_ops: SmallVec<[KleeneToken; 2]> = SmallVec::new();
            // We need to iterate from the end to start with outermost stack.
            for ops in acc.iter().rev() {
                occurrence_ops.extend_from_slice(ops);
            }
            ops_is_prefix(sess, node_id, span, name, &binder.ops, &occurrence_ops);
            return;
        }
    }
    buffer_lint(sess, span.into(), node_id, &format!("unknown macro variable `{}`", name));
}

/// Returns whether `binder_ops` is a prefix of `occurrence_ops`.
///
/// The stack of Kleene operators of a meta-variable occurrence just needs to have the stack of
/// Kleene operators of its binder as a prefix.
///
/// Consider $i in the following example:
///
///     ( $( $i:ident = $($j:ident),+ );* ) => { $($( $i += $j; )+)* }
///
/// It occurs under the Kleene stack ["*", "+"] and is bound under ["*"] only.
///
/// Arguments:
/// - `sess` is used to emit diagnostics and lints
/// - `node_id` is used to emit lints
/// - `span` is the span of the meta-variable being check
/// - `name` is the name of the meta-variable being check
/// - `binder_ops` is the stack of Kleene operators for the binder
/// - `occurrence_ops` is the stack of Kleene operators for the occurrence
fn ops_is_prefix(
    sess: &ParseSess,
    node_id: NodeId,
    span: Span,
    name: MacroRulesNormalizedIdent,
    binder_ops: &[KleeneToken],
    occurrence_ops: &[KleeneToken],
) {
    for (i, binder) in binder_ops.iter().enumerate() {
        if i >= occurrence_ops.len() {
            let mut span = MultiSpan::from_span(span);
            span.push_span_label(binder.span, "expected repetition".into());
            let message = &format!("variable '{}' is still repeating at this depth", name);
            buffer_lint(sess, span, node_id, message);
            return;
        }
        let occurrence = &occurrence_ops[i];
        if occurrence.op != binder.op {
            let mut span = MultiSpan::from_span(span);
            span.push_span_label(binder.span, "expected repetition".into());
            span.push_span_label(occurrence.span, "conflicting repetition".into());
            let message = "meta-variable repeats with different Kleene operator";
            buffer_lint(sess, span, node_id, message);
            return;
        }
    }
}

fn buffer_lint(sess: &ParseSess, span: MultiSpan, node_id: NodeId, message: &str) {
    // Macros loaded from other crates have dummy node ids.
    if node_id != DUMMY_NODE_ID {
        sess.buffer_lint(&META_VARIABLE_MISUSE, span, node_id, message);
    }
}
