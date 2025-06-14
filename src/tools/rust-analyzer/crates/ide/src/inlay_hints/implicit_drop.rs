//! Implementation of "implicit drop" inlay hints:
//! ```ignore
//! let x = vec![2];
//! if some_condition() {
//!     /* drop(x) */return;
//! }
//! ```
use hir::{
    ChalkTyInterner, DefWithBody,
    db::{DefDatabase as _, HirDatabase as _},
    mir::{MirSpan, TerminatorKind},
};
use ide_db::{FileRange, famous_defs::FamousDefs};

use syntax::{
    ToSmolStr,
    ast::{self, AstNode},
    match_ast,
};

use crate::{InlayHint, InlayHintLabel, InlayHintPosition, InlayHintsConfig, InlayKind};

pub(super) fn hints(
    acc: &mut Vec<InlayHint>,
    FamousDefs(sema, _): &FamousDefs<'_, '_>,
    config: &InlayHintsConfig,
    display_target: hir::DisplayTarget,
    node: &ast::Fn,
) -> Option<()> {
    if !config.implicit_drop_hints {
        return None;
    }

    let def = sema.to_def(node)?;
    let def: DefWithBody = def.into();

    let (hir, source_map) = sema.db.body_with_source_map(def.into());

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
            let Some(&binding_idx) = local_to_binding.get(place.local) else {
                continue; // Ignore temporary values
            };
            let range = match terminator.span {
                MirSpan::ExprId(e) => match source_map.expr_syntax(e) {
                    // don't show inlay hint for macro
                    Ok(s) if !s.file_id.is_macro() => {
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
                    _ => continue,
                },
                MirSpan::PatId(p) => match source_map.pat_syntax(p) {
                    Ok(s) if !s.file_id.is_macro() => s.value.text_range(),
                    _ => continue,
                },
                MirSpan::BindingId(b) => {
                    match source_map
                        .patterns_for_binding(b)
                        .iter()
                        .find_map(|p| source_map.pat_syntax(*p).ok())
                    {
                        Some(s) if !s.file_id.is_macro() => s.value.text_range(),
                        _ => continue,
                    }
                }
                MirSpan::SelfParam => match source_map.self_param_syntax() {
                    Some(s) if !s.file_id.is_macro() => s.value.text_range(),
                    _ => continue,
                },
                MirSpan::Unknown => continue,
            };
            let binding = &hir.bindings[binding_idx];
            let name = binding.name.display_no_db(display_target.edition).to_smolstr();
            if name.starts_with("<ra@") {
                continue; // Ignore desugared variables
            }
            let mut label = InlayHintLabel::simple(
                name,
                None,
                config.lazy_location_opt(|| {
                    source_map
                        .patterns_for_binding(binding_idx)
                        .first()
                        .and_then(|d| source_map.pat_syntax(*d).ok())
                        .and_then(|d| {
                            Some(FileRange {
                                file_id: d.file_id.file_id()?.file_id(sema.db),
                                range: d.value.text_range(),
                            })
                        })
                }),
            );
            label.prepend_str("drop(");
            label.append_str(")");
            acc.push(InlayHint {
                range,
                position: InlayHintPosition::After,
                pad_left: true,
                pad_right: true,
                kind: InlayKind::Drop,
                label,
                text_edit: None,
                resolve_parent: Some(node.syntax().text_range()),
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
        InlayHintsConfig,
        inlay_hints::tests::{DISABLED_CONFIG, check_with_config},
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

    #[test]
    fn ignore_inlay_hint_for_macro_call() {
        check_with_config(
            ONLY_DROP_CONFIG,
            r#"
    struct X;

    macro_rules! my_macro {
        () => {{
            let bbb = X;
            bbb
        }};
    }

    fn test() -> X {
        my_macro!()
    }
"#,
        );
    }
}
