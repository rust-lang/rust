//! Implementation of "implicit drop" inlay hints:
//! ```no_run
//! fn main() {
//!     let x = vec![2];
//!     if some_condition() {
//!         /* drop(x) */return;
//!     }
//! }
//! ```
use hir::{
    db::{DefDatabase as _, HirDatabase as _},
    mir::{MirSpan, TerminatorKind},
    ChalkTyInterner, DefWithBody, Semantics,
};
use ide_db::{base_db::FileRange, RootDatabase};

use syntax::{
    ast::{self, AstNode},
    match_ast,
};

use crate::{InlayHint, InlayHintLabel, InlayHintPosition, InlayHintsConfig, InlayKind};

pub(super) fn hints(
    acc: &mut Vec<InlayHint>,
    sema: &Semantics<'_, RootDatabase>,
    config: &InlayHintsConfig,
    def: &ast::Fn,
) -> Option<()> {
    if !config.implicit_drop_hints {
        return None;
    }

    let def = sema.to_def(def)?;
    let def: DefWithBody = def.into();

    let source_map = sema.db.body_with_source_map(def.into()).1;

    let hir = sema.db.body(def.into());
    let mir = sema.db.mir_body(def.into()).ok()?;

    let local_to_binding = mir.local_to_binding_map();

    for (_, bb) in mir.basic_blocks.iter() {
        let terminator = bb.terminator.as_ref()?;
        if let TerminatorKind::Drop { place, .. } = terminator.kind {
            if !place.projection.is_empty() {
                continue; // Ignore complex cases for now
            }
            if mir.locals[place.local].ty.adt_id(ChalkTyInterner).is_none() {
                continue; // Arguably only ADTs have significant drop impls
            }
            let Some(binding) = local_to_binding.get(place.local) else {
                continue; // Ignore temporary values
            };
            let range = match terminator.span {
                MirSpan::ExprId(e) => match source_map.expr_syntax(e) {
                    Ok(s) => {
                        let root = &s.file_syntax(sema.db);
                        let expr = s.value.to_node(root);
                        let expr = expr.syntax();
                        match_ast! {
                            match expr {
                                ast::BlockExpr(x) => x.stmt_list().and_then(|x| x.r_curly_token()).map(|x| x.text_range()).unwrap_or_else(|| expr.text_range()),
                                // make the inlay hint appear after the semicolon if there is
                                _ => {
                                    let nearest_semicolon = nearest_token_after_node(expr, syntax::SyntaxKind::SEMICOLON);
                                    nearest_semicolon.map(|x| x.text_range()).unwrap_or_else(|| expr.text_range())
                                },
                            }
                        }
                    }
                    Err(_) => continue,
                },
                MirSpan::PatId(p) => match source_map.pat_syntax(p) {
                    Ok(s) => s.value.text_range(),
                    Err(_) => continue,
                },
                MirSpan::Unknown => continue,
            };
            let binding = &hir.bindings[*binding];
            let binding_source = binding
                .definitions
                .first()
                .and_then(|d| source_map.pat_syntax(*d).ok())
                .and_then(|d| {
                    Some(FileRange { file_id: d.file_id.file_id()?, range: d.value.text_range() })
                });
            let name = binding.name.to_smol_str();
            if name.starts_with("<ra@") {
                continue; // Ignore desugared variables
            }
            let mut label = InlayHintLabel::simple(
                name,
                Some(crate::InlayTooltip::String("moz".into())),
                binding_source,
            );
            label.prepend_str("drop(");
            label.append_str(")");
            acc.push(InlayHint {
                range,
                position: InlayHintPosition::After,
                pad_left: true,
                pad_right: true,
                kind: InlayKind::Drop,
                needs_resolve: label.needs_resolve(),
                label,
                text_edit: None,
            })
        }
    }

    Some(())
}

fn nearest_token_after_node(
    node: &syntax::SyntaxNode,
    token_type: syntax::SyntaxKind,
) -> Option<syntax::SyntaxToken> {
    node.siblings_with_tokens(syntax::Direction::Next)
        .filter_map(|it| it.as_token().cloned())
        .find(|it| it.kind() == token_type)
}

#[cfg(test)]
mod tests {
    use crate::{
        inlay_hints::tests::{check_with_config, DISABLED_CONFIG},
        InlayHintsConfig,
    };

    const ONLY_DROP_CONFIG: InlayHintsConfig =
        InlayHintsConfig { implicit_drop_hints: true, ..DISABLED_CONFIG };

    #[test]
    fn basic() {
        check_with_config(
            ONLY_DROP_CONFIG,
            r#"
    struct X;
    fn f() {
        let x = X;
        if 2 == 5 {
            return;
                //^ drop(x)
        }
    }
  //^ drop(x)
"#,
        );
    }

    #[test]
    fn no_hint_for_copy_types_and_mutable_references() {
        // `T: Copy` and `T = &mut U` types do nothing on drop, so we should hide drop inlay hint for them.
        check_with_config(
            ONLY_DROP_CONFIG,
            r#"
//- minicore: copy, derive

    struct X(i32, i32);
    #[derive(Clone, Copy)]
    struct Y(i32, i32);
    fn f() {
        let a = 2;
        let b = a + 4;
        let mut x = X(a, b);
        let mut y = Y(a, b);
        let mx = &mut x;
        let my = &mut y;
        let c = a + b;
    }
  //^ drop(x)
"#,
        );
    }

    #[test]
    fn try_operator() {
        // We currently show drop inlay hint for every `?` operator that may potentially drop something. We probably need to
        // make it configurable as it doesn't seem very useful.
        check_with_config(
            ONLY_DROP_CONFIG,
            r#"
//- minicore: copy, try, option

    struct X;
    fn f() -> Option<()> {
        let x = X;
        let t_opt = Some(2);
        let t = t_opt?;
                    //^ drop(x)
        Some(())
    }
  //^ drop(x)
"#,
        );
    }

    #[test]
    fn if_let() {
        check_with_config(
            ONLY_DROP_CONFIG,
            r#"
    struct X;
    fn f() {
        let x = X;
        if let X = x {
            let y = X;
        }
      //^ drop(y)
    }
  //^ drop(x)
"#,
        );
    }
}
