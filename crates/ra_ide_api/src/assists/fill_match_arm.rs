use std::fmt::Write;
use hir::{
    AdtDef,
    source_binder,
    Ty,
    FieldSource,
};
use ra_ide_api_light::{
    assists::{
        Assist,
        AssistBuilder
    }
};
use ra_syntax::{
    ast::{
        self,
        AstNode,
    }
};

use crate::assists::AssistCtx;

pub fn fill_match_arm(ctx: AssistCtx) -> Option<Assist> {
    let match_expr = ctx.node_at_offset::<ast::MatchExpr>()?;

    // We already have some match arms, so we don't provide any assists.
    match match_expr.match_arm_list() {
        Some(arm_list) if arm_list.arms().count() > 0 => {
            return None;
        }
        _ => {}
    }

    let expr = match_expr.expr()?;
    let function = source_binder::function_from_child_node(ctx.db, ctx.file_id, expr.syntax())?;
    let infer_result = function.infer(ctx.db);
    let syntax_mapping = function.body_syntax_mapping(ctx.db);
    let node_expr = syntax_mapping.node_expr(expr)?;
    let match_expr_ty = infer_result[node_expr].clone();
    match match_expr_ty {
        Ty::Adt { def_id, .. } => match def_id {
            AdtDef::Enum(e) => {
                let mut buf = format!("match {} {{\n", expr.syntax().text().to_string());
                let variants = e.variants(ctx.db);
                for variant in variants {
                    let name = variant.name(ctx.db)?;
                    write!(
                        &mut buf,
                        "    {}::{}",
                        e.name(ctx.db)?.to_string(),
                        name.to_string()
                    )
                    .expect("write fmt");

                    let pat = variant
                        .fields(ctx.db)
                        .into_iter()
                        .map(|field| {
                            let name = field.name(ctx.db).to_string();
                            let (_, source) = field.source(ctx.db);
                            match source {
                                FieldSource::Named(_) => name,
                                FieldSource::Pos(_) => "_".to_string(),
                            }
                        })
                        .collect::<Vec<_>>();

                    match pat.first().map(|s| s.as_str()) {
                        Some("_") => write!(&mut buf, "({})", pat.join(", ")).expect("write fmt"),
                        Some(_) => write!(&mut buf, "{{{}}}", pat.join(", ")).expect("write fmt"),
                        None => (),
                    };

                    buf.push_str(" => (),\n");
                }
                buf.push_str("}");
                ctx.build("fill match arms", |edit: &mut AssistBuilder| {
                    edit.replace_node_and_indent(match_expr.syntax(), buf);
                })
            }
            _ => None,
        },
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use insta::assert_debug_snapshot_matches;

    use ra_syntax::{TextRange, TextUnit};

    use crate::{
        FileRange,
        mock_analysis::{analysis_and_position, single_file_with_position}
};
    use ra_db::SourceDatabase;

    fn test_assit(name: &str, code: &str) {
        let (analysis, position) = if code.contains("//-") {
            analysis_and_position(code)
        } else {
            single_file_with_position(code)
        };
        let frange = FileRange {
            file_id: position.file_id,
            range: TextRange::offset_len(position.offset, TextUnit::from(1)),
        };
        let source_file = analysis
            .with_db(|db| db.parse(frange.file_id))
            .expect("source file");
        let ret = analysis
            .with_db(|db| crate::assists::assists(db, frange.file_id, &source_file, frange.range))
            .expect("assists");

        assert_debug_snapshot_matches!(name, ret);
    }

    #[test]
    fn test_fill_match_arm() {
        test_assit(
            "fill_match_arm1",
            r#"
        enum A {
            As,
            Bs,
            Cs(String),
            Ds(String, String),
            Es{x: usize, y: usize}
        }

        fn main() {
            let a = A::As;
            match a<|>
        }
        "#,
        );

        test_assit(
            "fill_match_arm2",
            r#"
        enum A {
            As,
            Bs,
            Cs(String),
            Ds(String, String),
            Es{x: usize, y: usize}
        }

        fn main() {
            let a = A::As;
            match a<|> {}
        }
        "#,
        );
    }
}
