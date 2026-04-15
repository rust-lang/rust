use syntax::T;
use syntax::ast::RangeItem;
use syntax::ast::edit::AstNodeEdit;
use syntax::ast::{self, AstNode, HasName, LetStmt, Pat};
use syntax::syntax_editor::SyntaxEditor;

use crate::{AssistContext, AssistId, Assists};

// Assist: convert_let_else_to_match
//
// Converts let-else statement to let statement and match expression.
//
// ```
// fn main() {
//     let Ok(mut x) = f() else$0 { return };
// }
// ```
// ->
// ```
// fn main() {
//     let mut x = match f() {
//         Ok(x) => x,
//         _ => return,
//     };
// }
// ```
pub(crate) fn convert_let_else_to_match(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let (mut editor, _) = SyntaxEditor::new(ctx.source_file().syntax().clone());
    // Should focus on the `else` token to trigger
    let let_stmt = ctx
        .find_token_syntax_at_offset(T![else])
        .and_then(|it| it.parent()?.parent())
        .or_else(|| ctx.find_token_syntax_at_offset(T![let])?.parent())?;
    let let_stmt = LetStmt::cast(let_stmt)?;
    let else_block = let_stmt.let_else()?.block_expr()?;
    let else_expr = if else_block.statements().next().is_none()
        && let Some(tail_expr) = else_block.tail_expr()
    {
        tail_expr.reset_indent()
    } else {
        else_block.reset_indent().into()
    };
    let init = let_stmt.initializer()?;
    // Ignore let stmt with type annotation
    if let_stmt.ty().is_some() {
        return None;
    }
    let pat = let_stmt.pat()?;
    let mut idents = Vec::default();
    let pat_without_mut = remove_mut_and_collect_idents(&mut editor, &pat, &mut idents)?;
    let bindings = idents
        .into_iter()
        .filter_map(|ref pat| {
            // Identifiers which resolve to constants are not bindings
            if ctx.sema.resolve_bind_pat_to_const(pat).is_none() {
                Some((pat.name()?, pat.ref_token().is_none() && pat.mut_token().is_some()))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    acc.add(
        AssistId::refactor_rewrite("convert_let_else_to_match"),
        if bindings.is_empty() {
            "Convert let-else to match"
        } else {
            "Convert let-else to let and match"
        },
        let_stmt.syntax().text_range(),
        |builder| {
            let binding_paths = bindings
                .iter()
                .map(|(name, _)| {
                    editor.make().expr_path(editor.make().ident_path(&name.to_string()))
                })
                .collect::<Vec<_>>();

            let binding_arm = editor.make().match_arm(
                pat_without_mut,
                None,
                // There are three possible cases:
                //
                // - No bindings: `None => {}`
                // - Single binding: `Some(it) => it`
                // - Multiple bindings: `Foo::Bar { a, b, .. } => (a, b)`
                match binding_paths.len() {
                    0 => editor.make().expr_empty_block().into(),

                    1 => binding_paths[0].clone(),
                    _ => editor.make().expr_tuple(binding_paths).into(),
                },
            );
            let else_arm =
                editor.make().match_arm(editor.make().wildcard_pat().into(), None, else_expr);
            let arms = [binding_arm, else_arm].map(|arm| arm.indent(1.into()));
            let match_ = editor.make().expr_match(init, editor.make().match_arm_list(arms));
            let match_ = match_.indent(let_stmt.indent_level());

            if bindings.is_empty() {
                editor.replace(let_stmt.syntax(), match_.syntax());
            } else {
                let ident_pats = bindings
                    .into_iter()
                    .map(|(name, is_mut)| editor.make().ident_pat(false, is_mut, name).into())
                    .collect::<Vec<Pat>>();
                let new_let_stmt = editor.make().let_stmt(
                    if ident_pats.len() == 1 {
                        ident_pats[0].clone()
                    } else {
                        editor.make().tuple_pat(ident_pats).into()
                    },
                    None,
                    Some(match_.into()),
                );
                editor.replace(let_stmt.syntax(), new_let_stmt.syntax());
            }
            builder.add_file_edits(ctx.vfs_file_id(), editor);
        },
    )
}

fn remove_mut_and_collect_idents(
    editor: &mut SyntaxEditor,
    pat: &ast::Pat,
    acc: &mut Vec<ast::IdentPat>,
) -> Option<ast::Pat> {
    Some(match pat {
        ast::Pat::IdentPat(p) => {
            acc.push(p.clone());
            let non_mut_pat = editor.make().ident_pat(
                p.ref_token().is_some(),
                p.ref_token().is_some() && p.mut_token().is_some(),
                p.name()?,
            );
            let non_mut_pat = if let Some(inner) = p.pat() {
                non_mut_pat.set_pat(remove_mut_and_collect_idents(editor, &inner, acc), editor)
            } else {
                non_mut_pat
            };
            non_mut_pat.into()
        }
        ast::Pat::BoxPat(p) => {
            let pat = remove_mut_and_collect_idents(editor, &p.pat()?, acc)?;
            editor.make().box_pat(pat).into()
        }
        ast::Pat::OrPat(p) => {
            let pats = p
                .pats()
                .map(|pat| remove_mut_and_collect_idents(editor, &pat, acc))
                .collect::<Option<Vec<_>>>()?;
            editor.make().or_pat(pats, p.leading_pipe().is_some()).into()
        }
        ast::Pat::ParenPat(p) => {
            let pat = remove_mut_and_collect_idents(editor, &p.pat()?, acc)?;
            editor.make().paren_pat(pat).into()
        }
        ast::Pat::RangePat(p) => {
            let start = if let Some(start) = p.start() {
                Some(remove_mut_and_collect_idents(editor, &start, acc)?)
            } else {
                None
            };
            let end = if let Some(end) = p.end() {
                Some(remove_mut_and_collect_idents(editor, &end, acc)?)
            } else {
                None
            };
            editor.make().range_pat(start, end).into()
        }
        ast::Pat::RecordPat(p) => {
            let fields = p
                .record_pat_field_list()?
                .fields()
                .map(|field| {
                    remove_mut_and_collect_idents(editor, &field.pat()?, acc).map(|pat| {
                        if let Some(name_ref) = field.name_ref() {
                            editor.make().record_pat_field(name_ref, pat)
                        } else {
                            editor.make().record_pat_field_shorthand(pat)
                        }
                    })
                })
                .collect::<Option<Vec<_>>>()?;
            editor
                .make()
                .record_pat_with_fields(
                    p.path()?,
                    editor
                        .make()
                        .record_pat_field_list(fields, p.record_pat_field_list()?.rest_pat()),
                )
                .into()
        }
        ast::Pat::RefPat(p) => {
            let inner = p.pat()?;
            if let ast::Pat::IdentPat(ident) = inner {
                acc.push(ident);
                p.clone().into()
            } else {
                let pat = remove_mut_and_collect_idents(editor, &inner, acc)?;
                editor.make().ref_pat(pat).into()
            }
        }
        ast::Pat::SlicePat(p) => {
            let pats = p
                .pats()
                .map(|pat| remove_mut_and_collect_idents(editor, &pat, acc))
                .collect::<Option<Vec<_>>>()?;
            editor.make().slice_pat(pats).into()
        }
        ast::Pat::TuplePat(p) => {
            let pats = p
                .fields()
                .map(|field| remove_mut_and_collect_idents(editor, &field, acc))
                .collect::<Option<Vec<_>>>()?;
            editor.make().tuple_pat(pats).into()
        }
        ast::Pat::TupleStructPat(p) => {
            let fields = p
                .fields()
                .map(|field| remove_mut_and_collect_idents(editor, &field, acc))
                .collect::<Option<Vec<_>>>()?;
            editor.make().tuple_struct_pat(p.path()?, fields).into()
        }
        ast::Pat::RestPat(_)
        | ast::Pat::LiteralPat(_)
        | ast::Pat::PathPat(_)
        | ast::Pat::WildcardPat(_)
        | ast::Pat::ConstBlockPat(_) => pat.clone(),
        // don't support macro pat yet
        ast::Pat::MacroPat(_) => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::tests::{check_assist, check_assist_not_applicable, check_assist_target};

    #[test]
    fn convert_let_else_to_match_no_type_let() {
        check_assist_not_applicable(
            convert_let_else_to_match,
            r#"
fn main() {
    let 1: u32 = v.iter().sum() else$0 { return };
}"#,
        );
    }

    #[test]
    fn convert_let_else_to_match_on_else() {
        check_assist_not_applicable(
            convert_let_else_to_match,
            r#"
fn main() {
    let Ok(x) = f() else {$0 return };
}
            "#,
        );
    }

    #[test]
    fn convert_let_else_to_match_no_macropat() {
        check_assist_not_applicable(
            convert_let_else_to_match,
            r#"
fn main() {
    let m!() = g() else$0 { return };
}
            "#,
        );
    }

    #[test]
    fn convert_let_else_to_match_target() {
        check_assist_target(
            convert_let_else_to_match,
            r"
fn main() {
    let Ok(x) = f() else$0 { continue };
}",
            "let Ok(x) = f() else { continue };",
        );
    }

    #[test]
    fn convert_let_else_to_match_basic() {
        check_assist(
            convert_let_else_to_match,
            r"
fn main() {
    let Ok(x) = f() else$0 { continue };
}",
            r"
fn main() {
    let x = match f() {
        Ok(x) => x,
        _ => continue,
    };
}",
        );
    }

    #[test]
    fn convert_let_else_to_match_with_empty_else_block() {
        check_assist(
            convert_let_else_to_match,
            r"
fn main() {
    let Ok(x) = f() else$0 {};
}",
            r"
fn main() {
    let x = match f() {
        Ok(x) => x,
        _ => {}
    };
}",
        );
    }

    #[test]
    fn convert_let_else_to_match_with_some_indent() {
        check_assist(
            convert_let_else_to_match,
            r#"
mod indent {
    fn main() {
        let Ok(x) = f() else$0 {
            log();
            unreachable!(
                "..."
            );
        };
    }
}"#,
            r#"
mod indent {
    fn main() {
        let x = match f() {
            Ok(x) => x,
            _ => {
                log();
                unreachable!(
                    "..."
                );
            }
        };
    }
}"#,
        );

        check_assist(
            convert_let_else_to_match,
            r#"
mod indent {
    fn main() {
        let Ok(x) = f() else$0 {
            unreachable!(
                "..."
            )
        };
    }
}"#,
            r#"
mod indent {
    fn main() {
        let x = match f() {
            Ok(x) => x,
            _ => unreachable!(
                "..."
            ),
        };
    }
}"#,
        );
    }

    #[test]
    fn convert_let_else_to_match_const_ref() {
        check_assist(
            convert_let_else_to_match,
            r"
enum Option<T> {
    Some(T),
    None,
}
use Option::*;
fn main() {
    let None = f() el$0se { continue };
}",
            r"
enum Option<T> {
    Some(T),
    None,
}
use Option::*;
fn main() {
    match f() {
        None => {}
        _ => continue,
    }
}",
        );
    }

    #[test]
    fn convert_let_else_to_match_const_ref_const() {
        check_assist(
            convert_let_else_to_match,
            r"
const NEG1: i32 = -1;
fn main() {
    let NEG1 = f() el$0se { continue };
}",
            r"
const NEG1: i32 = -1;
fn main() {
    match f() {
        NEG1 => {}
        _ => continue,
    }
}",
        );
    }

    #[test]
    fn convert_let_else_to_match_mut() {
        check_assist(
            convert_let_else_to_match,
            r"
fn main() {
    let Ok(mut x) = f() el$0se { continue };
}",
            r"
fn main() {
    let mut x = match f() {
        Ok(x) => x,
        _ => continue,
    };
}",
        );
    }

    #[test]
    fn convert_let_else_to_match_multi_binders() {
        check_assist(
            convert_let_else_to_match,
            r#"
fn main() {
    let ControlFlow::Break((x, "tag", y, ..)) = f() else$0 { g(); return };
}"#,
            r#"
fn main() {
    let (x, y) = match f() {
        ControlFlow::Break((x, "tag", y, ..)) => (x, y),
        _ => { g(); return }
    };
}"#,
        );
    }

    #[test]
    fn convert_let_else_to_match_slice() {
        check_assist(
            convert_let_else_to_match,
            r#"
fn main() {
    let [one, 1001, other] = f() else$0 { break };
}"#,
            r#"
fn main() {
    let (one, other) = match f() {
        [one, 1001, other] => (one, other),
        _ => break,
    };
}"#,
        );
    }

    #[test]
    fn convert_let_else_to_match_struct() {
        check_assist(
            convert_let_else_to_match,
            r#"
fn main() {
    let [Struct { inner: Some(it) }, 1001, other] = f() else$0 { break };
}"#,
            r#"
fn main() {
    let (it, other) = match f() {
        [Struct { inner: Some(it) }, 1001, other] => (it, other),
        _ => break,
    };
}"#,
        );
    }

    #[test]
    fn convert_let_else_to_match_struct_ident_pat() {
        check_assist(
            convert_let_else_to_match,
            r#"
fn main() {
    let [Struct { inner }, 1001, other] = f() else$0 { break };
}"#,
            r#"
fn main() {
    let (inner, other) = match f() {
        [Struct { inner }, 1001, other] => (inner, other),
        _ => break,
    };
}"#,
        );
    }

    #[test]
    fn convert_let_else_to_match_no_binder() {
        check_assist(
            convert_let_else_to_match,
            r#"
fn main() {
    let (8 | 9) = f() else$0 { panic!() };
}"#,
            r#"
fn main() {
    match f() {
        (8 | 9) => {}
        _ => panic!(),
    }
}"#,
        );
    }

    #[test]
    fn convert_let_else_to_match_range() {
        check_assist(
            convert_let_else_to_match,
            r#"
fn main() {
    let 1.. = f() e$0lse { return };
}"#,
            r#"
fn main() {
    match f() {
        1.. => {}
        _ => return,
    }
}"#,
        );
    }

    #[test]
    fn convert_let_else_to_match_refpat() {
        check_assist(
            convert_let_else_to_match,
            r#"
fn main() {
    let Ok(&mut x) = f(&mut 0) else$0 { return };
}"#,
            r#"
fn main() {
    let x = match f(&mut 0) {
        Ok(&mut x) => x,
        _ => return,
    };
}"#,
        );
    }

    #[test]
    fn convert_let_else_to_match_refmut() {
        check_assist(
            convert_let_else_to_match,
            r#"
fn main() {
    let Ok(ref mut x) = f() else$0 { return };
}"#,
            r#"
fn main() {
    let x = match f() {
        Ok(ref mut x) => x,
        _ => return,
    };
}"#,
        );
    }

    #[test]
    fn convert_let_else_to_match_atpat() {
        check_assist(
            convert_let_else_to_match,
            r#"
fn main() {
    let out @ Ok(ins) = f() else$0 { return };
}"#,
            r#"
fn main() {
    let (out, ins) = match f() {
        out @ Ok(ins) => (out, ins),
        _ => return,
    };
}"#,
        );
    }

    #[test]
    fn convert_let_else_to_match_complex_init() {
        check_assist(
            convert_let_else_to_match,
            r#"
fn main() {
    let v = vec![1, 2, 3];
    let &[mut x, y, ..] = &v.iter().collect::<Vec<_>>()[..] else$0 { return };
}"#,
            r#"
fn main() {
    let v = vec![1, 2, 3];
    let (mut x, y) = match &v.iter().collect::<Vec<_>>()[..] {
        &[x, y, ..] => (x, y),
        _ => return,
    };
}"#,
        );
    }
}
