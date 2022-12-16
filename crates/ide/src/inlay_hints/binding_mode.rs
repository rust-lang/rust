use hir::{Mutability, Semantics};
use ide_db::RootDatabase;

use syntax::ast::{self, AstNode};

use crate::{InlayHint, InlayHintsConfig, InlayKind, InlayTooltip};

pub(super) fn hints(
    acc: &mut Vec<InlayHint>,
    sema: &Semantics<'_, RootDatabase>,
    config: &InlayHintsConfig,
    pat: &ast::Pat,
) -> Option<()> {
    if !config.binding_mode_hints {
        return None;
    }

    let outer_paren_pat = pat
        .syntax()
        .ancestors()
        .skip(1)
        .map_while(ast::Pat::cast)
        .map_while(|pat| match pat {
            ast::Pat::ParenPat(pat) => Some(pat),
            _ => None,
        })
        .last();
    let range =
        outer_paren_pat.as_ref().map_or_else(|| pat.syntax(), |it| it.syntax()).text_range();
    sema.pattern_adjustments(&pat).iter().for_each(|ty| {
        let reference = ty.is_reference();
        let mut_reference = ty.is_mutable_reference();
        let r = match (reference, mut_reference) {
            (true, true) => "&mut",
            (true, false) => "&",
            _ => return,
        };
        acc.push(InlayHint {
            range,
            kind: InlayKind::BindingModeHint,
            label: r.to_string().into(),
            tooltip: Some(InlayTooltip::String("Inferred binding mode".into())),
        });
    });
    match pat {
        ast::Pat::IdentPat(pat) if pat.ref_token().is_none() && pat.mut_token().is_none() => {
            let bm = sema.binding_mode_of_pat(pat)?;
            let bm = match bm {
                hir::BindingMode::Move => return None,
                hir::BindingMode::Ref(Mutability::Mut) => "ref mut",
                hir::BindingMode::Ref(Mutability::Shared) => "ref",
            };
            acc.push(InlayHint {
                range: pat.syntax().text_range(),
                kind: InlayKind::BindingModeHint,
                label: bm.to_string().into(),
                tooltip: Some(InlayTooltip::String("Inferred binding mode".into())),
            });
        }
        ast::Pat::OrPat(pat) if outer_paren_pat.is_none() => {
            acc.push(InlayHint {
                range: pat.syntax().text_range(),
                kind: InlayKind::OpeningParenthesis,
                label: "(".into(),
                tooltip: None,
            });
            acc.push(InlayHint {
                range: pat.syntax().text_range(),
                kind: InlayKind::ClosingParenthesis,
                label: ")".into(),
                tooltip: None,
            });
        }
        _ => (),
    }

    Some(())
}
