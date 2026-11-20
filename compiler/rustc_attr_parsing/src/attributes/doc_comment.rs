use std::mem;

use rustc_ast::AttrKind;
use rustc_ast::token::CommentKind;
use rustc_ast::visit::AssocCtxt;
use rustc_attr_ir::target::{AstTarget, MethodKind, Target};
use rustc_errors::{Diagnostic, MultiSpan};
use rustc_lint_defs::LintId;
use rustc_lint_defs::builtin::UNUSED_DOC_COMMENTS;
use rustc_span::Span;

use crate::attributes::prelude::{Allow, Error};
use crate::diagnostics::{BuiltinUnusedDocComment, BuiltinUnusedDocCommentSub};
use crate::target_checking::Policy::AllowSilent;
use crate::target_checking::{AllowedResult, AllowedTargets};
use crate::{AttributeParser, EmitAttribute, ShouldEmit};

pub(crate) fn target_check_doc_comment(
    parser: &AttributeParser<'_>,
    attr: &rustc_ast::Attribute,
    target: &AstTarget<'_>,
    target_span: Span,
    emit_lint: &mut impl FnMut(LintId, MultiSpan, EmitAttribute),
    has_target_checked_doc_comment: &mut bool,
) {
    if matches!(parser.should_emit, ShouldEmit::Nothing) {
        return;
    }
    if mem::replace(has_target_checked_doc_comment, true) {
        return;
    }
    let target = Target::from_ast_target(target);
    let res = DOC_TARGETS.is_allowed(target);
    if res == AllowedResult::Allowed {
        return;
    }

    let sub = match attr.kind {
        AttrKind::DocComment(CommentKind::Line, _) | AttrKind::Normal(..) => {
            BuiltinUnusedDocCommentSub::PlainHelp
        }
        AttrKind::DocComment(CommentKind::Block, _) => BuiltinUnusedDocCommentSub::BlockHelp,
        AttrKind::Synthetic(..) | AttrKind::Generated(..) => unreachable!(),
    };

    let context =
        if attr.span.ctxt() == target_span.ctxt() { target_span.shrink_to_lo() } else { attr.span };

    let attr_span = attr.span;
    emit_lint(
        LintId::of(UNUSED_DOC_COMMENTS),
        attr_span.into(),
        EmitAttribute(Box::new(move |dcx, level, _| {
            BuiltinUnusedDocComment { sub, target: target.plural_name(), label: attr_span, context }
                .into_diag(dcx, level)
        })),
    );
}

/// This set of targets is used both for the `doc` attribute and doc comments.
pub(crate) const DOC_TARGETS: AllowedTargets<'static> = {
    AllowedTargets::AllowListWarnRest(&[
        Allow(Target::Use),
        Allow(Target::Static),
        Allow(Target::Const),
        Allow(Target::Fn),
        Allow(Target::Mod),
        Allow(Target::Enum),
        Allow(Target::Variant),
        Allow(Target::Struct),
        Allow(Target::Field),
        Allow(Target::Union),
        Allow(Target::Trait),
        Allow(Target::TraitAlias),
        Allow(Target::TyAlias),
        Allow(Target::AssocConst(AssocCtxt::Impl { of_trait: false })),
        Allow(Target::AssocConst(AssocCtxt::Trait)),
        Allow(Target::AssocConst(AssocCtxt::Impl { of_trait: true })),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Method(MethodKind::Trait { body: false })),
        Allow(Target::Method(MethodKind::Trait { body: true })),
        Allow(Target::Method(MethodKind::TraitImpl)),
        Allow(Target::AssocTy(AssocCtxt::Impl { of_trait: false })),
        Allow(Target::AssocTy(AssocCtxt::Trait)),
        Allow(Target::AssocTy(AssocCtxt::Impl { of_trait: true })),
        Allow(Target::ForeignFn),
        Allow(Target::ForeignStatic),
        Allow(Target::ForeignTy),
        Allow(Target::MacroDef),
        Allow(Target::Crate),
        Allow(Target::ExternCrate),
        Allow(Target::Impl { of_trait: false }),
        Allow(Target::Impl { of_trait: true }),
        AllowSilent(Target::WherePredicate), // Error, checked elsewhere
        Error(Target::Param),
    ])
};
