use crate::{Assist, AssistCtx, AssistId, TextRange, TextUnit};
use hir::db::HirDatabase;
use ra_syntax::ast::{AstNode, MatchArm};

pub(crate) fn merge_match_arms(mut ctx: AssistCtx<impl HirDatabase>) -> Option<Assist> {
    let current_arm = ctx.node_at_offset::<MatchArm>()?;

    // We check if the following match arm matches this one. We could, but don't,
    // compare to the previous match arm as well.
    let next = current_arm.syntax().next_sibling();
    let next_arm = MatchArm::cast(next?.clone())?;

    // Don't try to handle arms with guards for now - can add support for this later
    if current_arm.guard().is_some() || next_arm.guard().is_some() {
        return None;
    }

    let current_expr = current_arm.expr()?;
    let next_expr = next_arm.expr()?;

    // Check for match arm equality by comparing lengths and then string contents
    if current_expr.syntax().text_range().len() != next_expr.syntax().text_range().len() {
        return None;
    }
    if current_expr.syntax().text() != next_expr.syntax().text() {
        return None;
    }

    let cursor_to_end = current_arm.syntax().text_range().end() - ctx.frange.range.start();

    ctx.add_action(AssistId("merge_match_arms"), "merge match arms", |edit| {
        fn contains_placeholder(a: &MatchArm) -> bool {
            a.pats().any(|x| match x.kind() {
                ra_syntax::ast::PatKind::PlaceholderPat(..) => true,
                _ => false,
            })
        }

        let pats = if contains_placeholder(&current_arm) || contains_placeholder(&next_arm) {
            "_".into()
        } else {
            let ps: Vec<String> = current_arm
                .pats()
                .map(|x| x.syntax().to_string())
                .chain(next_arm.pats().map(|x| x.syntax().to_string()))
                .collect();
            ps.join(" | ")
        };

        let arm = format!("{} => {}", pats, current_expr.syntax().text());
        let offset = TextUnit::from_usize(arm.len()) - cursor_to_end;

        let start = current_arm.syntax().text_range().start();
        let end = next_arm.syntax().text_range().end();

        edit.target(current_arm.syntax().text_range());
        edit.replace(TextRange::from_to(start, end), arm);
        edit.set_cursor(start + offset);
    });

    ctx.build()
}

#[cfg(test)]
mod tests {
    use super::merge_match_arms;
    use crate::helpers::{check_assist, check_assist_not_applicable};

    #[test]
    fn merge_match_arms_single_patterns() {
        check_assist(
            merge_match_arms,
            r#"
            #[derive(Debug)]
            enum X { A, B, C }

            fn main() {
                let x = X::A;
                let y = match x {
                    X::A => { 1i32<|> }
                    X::B => { 1i32 }
                    X::C => { 2i32 }
                }
            }
            "#,
            r#"
            #[derive(Debug)]
            enum X { A, B, C }

            fn main() {
                let x = X::A;
                let y = match x {
                    X::A | X::B => { 1i32<|> }
                    X::C => { 2i32 }
                }
            }
            "#,
        );
    }

    #[test]
    fn merge_match_arms_multiple_patterns() {
        check_assist(
            merge_match_arms,
            r#"
            #[derive(Debug)]
            enum X { A, B, C, D, E }

            fn main() {
                let x = X::A;
                let y = match x {
                    X::A | X::B => {<|> 1i32 },
                    X::C | X::D => { 1i32 },
                    X::E => { 2i32 },
                }
            }
            "#,
            r#"
            #[derive(Debug)]
            enum X { A, B, C, D, E }

            fn main() {
                let x = X::A;
                let y = match x {
                    X::A | X::B | X::C | X::D => {<|> 1i32 },
                    X::E => { 2i32 },
                }
            }
            "#,
        );
    }

    #[test]
    fn merge_match_arms_placeholder_pattern() {
        check_assist(
            merge_match_arms,
            r#"
            #[derive(Debug)]
            enum X { A, B, C, D, E }

            fn main() {
                let x = X::A;
                let y = match x {
                    X::A => { 1i32 },
                    X::B => { 2i<|>32 },
                    _ => { 2i32 }
                }
            }
            "#,
            r#"
            #[derive(Debug)]
            enum X { A, B, C, D, E }

            fn main() {
                let x = X::A;
                let y = match x {
                    X::A => { 1i32 },
                    _ => { 2i<|>32 }
                }
            }
            "#,
        );
    }

    #[test]
    fn merge_match_arms_rejects_guards() {
        check_assist_not_applicable(
            merge_match_arms,
            r#"
            #[derive(Debug)]
            enum X {
                A(i32),
                B,
                C
            }

            fn main() {
                let x = X::A;
                let y = match x {
                    X::A(a) if a > 5 => { <|>1i32 },
                    X::B => { 1i32 },
                    X::C => { 2i32 }
                }
            }
            "#,
        );
    }
}
