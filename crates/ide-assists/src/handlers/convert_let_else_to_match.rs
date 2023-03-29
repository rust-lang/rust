use hir::Semantics;
use ide_db::RootDatabase;
use syntax::ast::{edit::AstNodeEdit, AstNode, HasName, LetStmt, Name, Pat};
use syntax::T;

use crate::{AssistContext, AssistId, AssistKind, Assists};

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
    // should focus on else token to trigger
    let let_stmt = ctx
        .find_token_syntax_at_offset(T![else])
        .and_then(|it| it.parent()?.parent())
        .or_else(|| ctx.find_token_syntax_at_offset(T![let])?.parent())?;
    let let_stmt = LetStmt::cast(let_stmt)?;
    let let_else_block = let_stmt.let_else()?.block_expr()?;
    let let_init = let_stmt.initializer()?;
    if let_stmt.ty().is_some() {
        // don't support let with type annotation
        return None;
    }
    let pat = let_stmt.pat()?;
    let mut binders = Vec::new();
    binders_in_pat(&mut binders, &pat, &ctx.sema)?;

    let target = let_stmt.syntax().text_range();
    acc.add(
        AssistId("convert_let_else_to_match", AssistKind::RefactorRewrite),
        "Convert let-else to let and match",
        target,
        |edit| {
            let indent_level = let_stmt.indent_level().0 as usize;
            let indent = "    ".repeat(indent_level);
            let indent1 = "    ".repeat(indent_level + 1);

            let binders_str = binders_to_str(&binders, false);
            let binders_str_mut = binders_to_str(&binders, true);

            let init_expr = let_init.syntax().text();
            let mut pat_no_mut = pat.syntax().text().to_string();
            // remove the mut from the pattern
            for (b, ismut) in binders.iter() {
                if *ismut {
                    pat_no_mut = pat_no_mut.replace(&format!("mut {b}"), &b.to_string());
                }
            }

            let only_expr = let_else_block.statements().next().is_none();
            let branch2 = match &let_else_block.tail_expr() {
                Some(tail) if only_expr => format!("{tail},"),
                _ => let_else_block.syntax().text().to_string(),
            };
            let replace = if binders.is_empty() {
                format!(
                    "match {init_expr} {{
{indent1}{pat_no_mut} => {binders_str}
{indent1}_ => {branch2}
{indent}}}"
                )
            } else {
                format!(
                    "let {binders_str_mut} = match {init_expr} {{
{indent1}{pat_no_mut} => {binders_str},
{indent1}_ => {branch2}
{indent}}};"
                )
            };
            edit.replace(target, replace);
        },
    )
}

/// Gets a list of binders in a pattern, and whether they are mut.
fn binders_in_pat(
    acc: &mut Vec<(Name, bool)>,
    pat: &Pat,
    sem: &Semantics<'_, RootDatabase>,
) -> Option<()> {
    use Pat::*;
    match pat {
        IdentPat(p) => {
            let ident = p.name()?;
            let ismut = p.ref_token().is_none() && p.mut_token().is_some();
            // check for const reference
            if sem.resolve_bind_pat_to_const(p).is_none() {
                acc.push((ident, ismut));
            }
            if let Some(inner) = p.pat() {
                binders_in_pat(acc, &inner, sem)?;
            }
            Some(())
        }
        BoxPat(p) => p.pat().and_then(|p| binders_in_pat(acc, &p, sem)),
        RestPat(_) | LiteralPat(_) | PathPat(_) | WildcardPat(_) | ConstBlockPat(_) => Some(()),
        OrPat(p) => {
            for p in p.pats() {
                binders_in_pat(acc, &p, sem)?;
            }
            Some(())
        }
        ParenPat(p) => p.pat().and_then(|p| binders_in_pat(acc, &p, sem)),
        RangePat(p) => {
            if let Some(st) = p.start() {
                binders_in_pat(acc, &st, sem)?
            }
            if let Some(ed) = p.end() {
                binders_in_pat(acc, &ed, sem)?
            }
            Some(())
        }
        RecordPat(p) => {
            for f in p.record_pat_field_list()?.fields() {
                let pat = f.pat()?;
                binders_in_pat(acc, &pat, sem)?;
            }
            Some(())
        }
        RefPat(p) => p.pat().and_then(|p| binders_in_pat(acc, &p, sem)),
        SlicePat(p) => {
            for p in p.pats() {
                binders_in_pat(acc, &p, sem)?;
            }
            Some(())
        }
        TuplePat(p) => {
            for p in p.fields() {
                binders_in_pat(acc, &p, sem)?;
            }
            Some(())
        }
        TupleStructPat(p) => {
            for p in p.fields() {
                binders_in_pat(acc, &p, sem)?;
            }
            Some(())
        }
        // don't support macro pat yet
        MacroPat(_) => None,
    }
}

fn binders_to_str(binders: &[(Name, bool)], addmut: bool) -> String {
    let vars = binders
        .iter()
        .map(
            |(ident, ismut)| {
                if *ismut && addmut {
                    format!("mut {ident}")
                } else {
                    ident.to_string()
                }
            },
        )
        .collect::<Vec<_>>()
        .join(", ");
    if binders.is_empty() {
        String::from("{}")
    } else if binders.len() == 1 {
        vars
    } else {
        format!("({vars})")
    }
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
