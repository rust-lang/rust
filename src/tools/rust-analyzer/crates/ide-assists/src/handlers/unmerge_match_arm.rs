use syntax::{
    Direction, SyntaxKind, T,
    ast::{self, AstNode, edit::IndentLevel, syntax_factory::SyntaxFactory},
    syntax_editor::{Element, Position},
};

use crate::{AssistContext, AssistId, Assists};

// Assist: unmerge_match_arm
//
// Splits the current match with a `|` pattern into two arms with identical bodies.
//
// ```
// enum Action { Move { distance: u32 }, Stop }
//
// fn handle(action: Action) {
//     match action {
//         Action::Move(..) $0| Action::Stop => foo(),
//     }
// }
// ```
// ->
// ```
// enum Action { Move { distance: u32 }, Stop }
//
// fn handle(action: Action) {
//     match action {
//         Action::Move(..) => foo(),
//         Action::Stop => foo(),
//     }
// }
// ```
pub(crate) fn unmerge_match_arm(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let pipe_token = ctx.find_token_syntax_at_offset(T![|])?;
    let or_pat = ast::OrPat::cast(pipe_token.parent()?)?;
    if or_pat.leading_pipe().is_some_and(|it| it == pipe_token) {
        return None;
    }
    let match_arm = ast::MatchArm::cast(or_pat.syntax().parent()?)?;
    let match_arm_body = match_arm.expr()?;

    // We don't need to check for leading pipe because it is directly under `MatchArm`
    // without `OrPat`.

    let new_parent = match_arm.syntax().parent()?;

    acc.add(
        AssistId::refactor_rewrite("unmerge_match_arm"),
        "Unmerge match arm",
        pipe_token.text_range(),
        |edit| {
            let make = SyntaxFactory::with_mappings();
            let mut editor = edit.make_editor(&new_parent);
            let pats_after = pipe_token
                .siblings_with_tokens(Direction::Next)
                .filter_map(|it| ast::Pat::cast(it.into_node()?))
                .collect::<Vec<_>>();
            // It is guaranteed that `pats_after` has at least one element
            let new_pat = if pats_after.len() == 1 {
                pats_after[0].clone()
            } else {
                make.or_pat(pats_after, or_pat.leading_pipe().is_some()).into()
            };
            let new_match_arm = make.match_arm(new_pat, match_arm.guard(), match_arm_body);
            let mut pipe_index = pipe_token.index();
            if pipe_token
                .prev_sibling_or_token()
                .is_some_and(|it| it.kind() == SyntaxKind::WHITESPACE)
            {
                pipe_index -= 1;
            }
            for child in or_pat
                .syntax()
                .children_with_tokens()
                .skip_while(|child| child.index() < pipe_index)
            {
                editor.delete(child.syntax_element());
            }

            let mut insert_after_old_arm = Vec::new();

            // A comma can be:
            //  - After the arm. In this case we always want to insert a comma after the newly
            //    inserted arm.
            //  - Missing after the arm, with no arms after. In this case we want to insert a
            //    comma before the newly inserted arm. It can not be necessary if there arm
            //    body is a block, but we don't bother to check that.
            //  - Missing after the arm with arms after, if the arm body is a block. In this case
            //    we don't want to insert a comma at all.
            let has_comma_after = match_arm.comma_token().is_some();
            if !has_comma_after && !match_arm.expr().unwrap().is_block_like() {
                insert_after_old_arm.push(make.token(T![,]).into());
            }

            let indent = IndentLevel::from_node(match_arm.syntax());
            insert_after_old_arm.push(make.whitespace(&format!("\n{indent}")).into());

            insert_after_old_arm.push(new_match_arm.syntax().clone().into());

            editor.insert_all(Position::after(match_arm.syntax()), insert_after_old_arm);
            editor.add_mappings(make.finish_with_mappings());
            edit.add_file_edits(ctx.vfs_file_id(), editor);
        },
    )
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn unmerge_match_arm_single_pipe() {
        check_assist(
            unmerge_match_arm,
            r#"
#[derive(Debug)]
enum X { A, B, C }

fn main() {
    let x = X::A;
    let y = match x {
        X::A $0| X::B => { 1i32 }
        X::C => { 2i32 }
    };
}
"#,
            r#"
#[derive(Debug)]
enum X { A, B, C }

fn main() {
    let x = X::A;
    let y = match x {
        X::A => { 1i32 }
        X::B => { 1i32 }
        X::C => { 2i32 }
    };
}
"#,
        );
    }

    #[test]
    fn unmerge_match_arm_guard() {
        check_assist(
            unmerge_match_arm,
            r#"
#[derive(Debug)]
enum X { A, B, C }

fn main() {
    let x = X::A;
    let y = match x {
        X::A $0| X::B if true => { 1i32 }
        _ => { 2i32 }
    };
}
"#,
            r#"
#[derive(Debug)]
enum X { A, B, C }

fn main() {
    let x = X::A;
    let y = match x {
        X::A if true => { 1i32 }
        X::B if true => { 1i32 }
        _ => { 2i32 }
    };
}
"#,
        );
    }

    #[test]
    fn unmerge_match_arm_leading_pipe() {
        check_assist_not_applicable(
            unmerge_match_arm,
            r#"

fn main() {
    let y = match 0 {
        |$0 0 => { 1i32 }
        1 => { 2i32 }
    };
}
"#,
        );
    }

    #[test]
    fn unmerge_match_arm_multiple_pipes() {
        check_assist(
            unmerge_match_arm,
            r#"
#[derive(Debug)]
enum X { A, B, C, D, E }

fn main() {
    let x = X::A;
    let y = match x {
        X::A | X::B |$0 X::C | X::D => 1i32,
        X::E => 2i32,
    };
}
"#,
            r#"
#[derive(Debug)]
enum X { A, B, C, D, E }

fn main() {
    let x = X::A;
    let y = match x {
        X::A | X::B => 1i32,
        X::C | X::D => 1i32,
        X::E => 2i32,
    };
}
"#,
        );
    }

    #[test]
    fn unmerge_match_arm_inserts_comma_if_required() {
        check_assist(
            unmerge_match_arm,
            r#"
#[derive(Debug)]
enum X { A, B }

fn main() {
    let x = X::A;
    let y = match x {
        X::A $0| X::B => 1i32
    };
}
"#,
            r#"
#[derive(Debug)]
enum X { A, B }

fn main() {
    let x = X::A;
    let y = match x {
        X::A => 1i32,
        X::B => 1i32,
    };
}
"#,
        );
    }

    #[test]
    fn unmerge_match_arm_inserts_comma_if_had_after() {
        check_assist(
            unmerge_match_arm,
            r#"
#[derive(Debug)]
enum X { A, B }

fn main() {
    let x = X::A;
    match x {
        X::A $0| X::B => {}
    }
}
"#,
            r#"
#[derive(Debug)]
enum X { A, B }

fn main() {
    let x = X::A;
    match x {
        X::A => {}
        X::B => {}
    }
}
"#,
        );
    }
}
