//! Late validation of attributes using context only available after HIR is built.
//!
//! Some attribute checks need information that isn't available during parsing (e.g. whether
//! the parent item is a trait impl, or the ABI of a function). This module provides:
//!
//! - **[`LateValidationContext`]**: A context struct built by the compiler pass that has access
//!   to HIR/tcx. It holds only plain data (e.g. `Target`, `parent_is_trait_impl`) so this crate
//!   stays free of `rustc_middle` and type-checking.
//!
//! - **Per-attribute validators**: Functions that take a reference to the context and return
//!   an optional "validation result" (e.g. a span to lint). The pass in `rustc_passes` builds
//!   the context, calls the validator, and emits the actual diagnostic/lint.
//!
//! This design keeps validation *logic* in the attribute crate while diagnostics stay in
//! the pass crate. It is part of the [check_attrs cleanup](https://github.com/rust-lang/rust/issues/153101).
//!
//! **Note:** Several of these checks ideally run with full *target* context during parsing, not in a
//! later pass. That needs plumbing from AST/HIR into the parser pipeline (see the issue linked
//! above). Until then, this module centralizes predicates that still must run after HIR is built.

use rustc_hir::Target;
use rustc_hir::attrs::diagnostic::Directive;
use rustc_span::{Span, Symbol};

/// When the attribute is on an expression, describes that expression for validation.
#[derive(Clone, Debug)]
pub struct ExprContext {
    pub node_span: Span,
    pub is_loop: bool,
    pub is_break: bool,
}

/// Context passed to late validators. Built in `rustc_passes` from HIR/tcx.
///
/// Extend this struct when moving checks that need more than `Target` (e.g. `fn_abi`, …).
#[derive(Clone, Debug)]
pub struct LateValidationContext {
    /// The syntactic target the attribute was applied to.
    pub target: Target,
    /// Whether the parent item is a trait impl (e.g. `impl SomeTrait for T`).
    /// Used e.g. for `#[deprecated]` on trait impl members.
    pub parent_is_trait_impl: bool,
    /// When `target` is `Expression`, this is set with the expression's span and kind.
    pub expr_context: Option<ExprContext>,
    /// When `target` is `Impl { of_trait: true }`, whether the impl is `const`.
    pub impl_is_const: Option<bool>,
    /// When `target` is `Impl`, whether this is a trait impl (vs inherent).
    pub impl_of_trait: Option<bool>,
    /// When `target` is `ForeignMod`, whether the extern block has ABI = Rust (link invalid).
    pub foreign_mod_abi_is_rust: Option<bool>,
    /// When `target` is `MacroDef`, whether the macro is a decl macro (macro 2.0).
    pub macro_export_is_decl_macro: Option<bool>,
}

/// Result of late validation for `#[deprecated]`: emit a lint when present.
#[derive(Clone, Debug)]
pub struct DeprecatedValidation {
    pub attr_span: Span,
}

/// Validates `#[deprecated]` in contexts where it has no effect (e.g. on trait impl members).
///
/// Returns `Some(...)` when the pass should emit `DeprecatedAnnotationHasNoEffect`.
/// `attr_span` is the span of the `#[deprecated]` attribute.
pub fn validate_deprecated(
    ctx: &LateValidationContext,
    attr_span: Span,
) -> Option<DeprecatedValidation> {
    let has_no_effect =
        matches!(ctx.target, Target::AssocConst | Target::Method(..) | Target::AssocTy)
            && ctx.parent_is_trait_impl;

    if has_no_effect { Some(DeprecatedValidation { attr_span }) } else { None }
}

/// Result of late validation for `#[loop_match]`: emit error when not on a loop.
#[derive(Clone, Debug)]
pub struct LoopMatchValidation {
    pub attr_span: Span,
    pub node_span: Span,
}

/// Validates `#[loop_match]`: must be applied to a loop expression.
pub fn validate_loop_match(
    ctx: &LateValidationContext,
    attr_span: Span,
) -> Option<LoopMatchValidation> {
    if !matches!(ctx.target, Target::Expression) {
        return None;
    }
    let Some(expr) = &ctx.expr_context else {
        return None;
    };
    if expr.is_loop {
        None
    } else {
        Some(LoopMatchValidation { attr_span, node_span: expr.node_span })
    }
}

/// Result of late validation for `#[const_continue]`: emit error when not on a break.
#[derive(Clone, Debug)]
pub struct ConstContinueValidation {
    pub attr_span: Span,
    pub node_span: Span,
}

/// Validates `#[const_continue]`: must be applied to a break expression.
pub fn validate_const_continue(
    ctx: &LateValidationContext,
    attr_span: Span,
) -> Option<ConstContinueValidation> {
    if !matches!(ctx.target, Target::Expression) {
        return None;
    }
    let Some(expr) = &ctx.expr_context else {
        return None;
    };
    if expr.is_break {
        None
    } else {
        Some(ConstContinueValidation { attr_span, node_span: expr.node_span })
    }
}

// --- diagnostic::on_unimplemented (target-only) ---

/// Result: emit lint when `#[diagnostic::on_unimplemented]` is not on a trait.
#[derive(Clone, Debug)]
pub struct OnUnimplementedValidation {
    pub attr_span: Span,
}

/// Validates that `#[diagnostic::on_unimplemented]` is only applied to trait definitions.
pub fn validate_diagnostic_on_unimplemented(
    ctx: &LateValidationContext,
    attr_span: Span,
) -> Option<OnUnimplementedValidation> {
    if matches!(ctx.target, Target::Trait) {
        None
    } else {
        Some(OnUnimplementedValidation { attr_span })
    }
}

/// Invokes `on_unknown` for each `{ident}` in `directive`'s format strings that is not a declared
/// type (or type-alias) generic parameter of the surrounding item. Lifetimes in the generic list
/// are ignored by the caller via `is_declared_type_param`.
///
/// The pass supplies `is_declared_type_param` using HIR; parsing cannot do this without the item.
pub fn for_each_unknown_diagnostic_format_param(
    directive: &Directive,
    is_declared_type_param: impl Fn(Symbol) -> bool,
    mut on_unknown: impl FnMut(Symbol, Span),
) {
    directive.visit_params(&mut |argument_name, span| {
        if !is_declared_type_param(argument_name) {
            on_unknown(argument_name, span);
        }
    });
}

// --- diagnostic::on_move (target-only; format literals use [`for_each_unknown_diagnostic_format_param`]) ---

/// Result: emit lint when `#[diagnostic::on_move]` is not on an ADT definition.
#[derive(Clone, Debug)]
pub struct OnMoveTargetValidation {
    pub attr_span: Span,
}

/// Validates that `#[diagnostic::on_move]` is only applied to enums, structs, or unions.
pub fn validate_diagnostic_on_move_target(
    ctx: &LateValidationContext,
    attr_span: Span,
) -> Option<OnMoveTargetValidation> {
    if matches!(ctx.target, Target::Enum | Target::Struct | Target::Union) {
        None
    } else {
        Some(OnMoveTargetValidation { attr_span })
    }
}

// --- diagnostic::on_const ---

/// What went wrong with `#[diagnostic::on_const]`.
#[derive(Clone, Debug)]
pub enum OnConstValidation {
    /// Not on a trait impl.
    WrongTarget { item_span: Span },
    /// On a const trait impl (only non-const allowed).
    ConstImpl { item_span: Span },
}

/// Validates `#[diagnostic::on_const]`: only on non-const trait impls.
pub fn validate_diagnostic_on_const(
    ctx: &LateValidationContext,
    _attr_span: Span,
    item_span: Span,
) -> Option<OnConstValidation> {
    if ctx.target == (Target::Impl { of_trait: true }) {
        match ctx.impl_is_const {
            Some(true) => Some(OnConstValidation::ConstImpl { item_span }),
            Some(false) => None,
            None => None, // e.g. foreign item, skip
        }
    } else {
        Some(OnConstValidation::WrongTarget { item_span })
    }
}

// --- diagnostic::do_not_recommend ---

/// Result: emit lint when `#[diagnostic::do_not_recommend]` is not on a trait impl.
#[derive(Clone, Debug)]
pub struct DoNotRecommendValidation {
    pub attr_span: Span,
}

/// Validates that `#[diagnostic::do_not_recommend]` is only on trait implementations.
pub fn validate_do_not_recommend(
    ctx: &LateValidationContext,
    attr_span: Span,
) -> Option<DoNotRecommendValidation> {
    let on_trait_impl =
        matches!(ctx.target, Target::Impl { of_trait: true }) && ctx.impl_of_trait == Some(true);
    if on_trait_impl { None } else { Some(DoNotRecommendValidation { attr_span }) }
}

// --- macro_export ---

/// Result: emit lint when `#[macro_export]` is on a decl macro (macro 2.0).
#[derive(Clone, Debug)]
pub struct MacroExportValidation {
    pub attr_span: Span,
}

/// Validates `#[macro_export]`: not allowed on decl macros.
pub fn validate_macro_export(
    ctx: &LateValidationContext,
    attr_span: Span,
) -> Option<MacroExportValidation> {
    if ctx.target == Target::MacroDef && ctx.macro_export_is_decl_macro == Some(true) {
        Some(MacroExportValidation { attr_span })
    } else {
        None
    }
}

// --- link ---

/// Result: emit lint when `#[link]` is used in the wrong place.
#[derive(Clone, Debug)]
pub struct LinkValidation {
    pub attr_span: Span,
    /// If wrong target, pass the item span for the diagnostic.
    pub wrong_target_span: Option<Span>,
}

/// Validates `#[link]`: only on foreign modules, and not on `extern "Rust"`.
pub fn validate_link(
    ctx: &LateValidationContext,
    attr_span: Span,
    target_span: Span,
) -> Option<LinkValidation> {
    let valid =
        matches!(ctx.target, Target::ForeignMod) && ctx.foreign_mod_abi_is_rust != Some(true);
    if valid {
        None
    } else {
        Some(LinkValidation {
            attr_span,
            wrong_target_span: (ctx.target != Target::ForeignMod).then_some(target_span),
        })
    }
}

// ---------------------------------------------------------------------------
// Further moves (issue #153101)
// ---------------------------------------------------------------------------
//
// See the issue for remaining `check_attr` checks (naked, non_exhaustive, cross-attribute rules,
// argument-dependent targets, etc.). `check_proc_macro` cannot move (needs type information).
