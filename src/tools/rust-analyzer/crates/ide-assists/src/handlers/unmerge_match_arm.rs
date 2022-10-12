use syntax::{
    algo::neighbor,
    ast::{self, edit::IndentLevel, make, AstNode},
    ted::{self, Position},
    Direction, SyntaxKind, T,
};

use crate::{AssistContext, AssistId, AssistKind, Assists};

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
    let or_pat = ast::OrPat::cast(pipe_token.parent()?)?.clone_for_update();
    let match_arm = ast::MatchArm::cast(or_pat.syntax().parent()?)?;
    let match_arm_body = match_arm.expr()?;

    // We don't need to check for leading pipe because it is directly under `MatchArm`
    // without `OrPat`.

    let new_parent = match_arm.syntax().parent()?;
    let old_parent_range = new_parent.text_range();

    acc.add(
        AssistId("unmerge_match_arm", AssistKind::RefactorRewrite),
        "Unmerge match arm",
        pipe_token.text_range(),
        |edit| {
            let pats_after = pipe_token
                .siblings_with_tokens(Direction::Next)
                .filter_map(|it| ast::Pat::cast(it.into_node()?));
            // FIXME: We should add a leading pipe if the original arm has one.
            let new_match_arm = make::match_arm(
                pats_after,
                match_arm.guard().and_then(|guard| guard.condition()),
                match_arm_body,
            )
            .clone_for_update();

            let mut pipe_index = pipe_token.index();
            if pipe_token
                .prev_sibling_or_token()
                .map_or(false, |it| it.kind() == SyntaxKind::WHITESPACE)
            {
                pipe_index -= 1;
            }
            or_pat.syntax().splice_children(
                pipe_index..or_pat.syntax().children_with_tokens().count(),
                Vec::new(),
            );

            let mut insert_after_old_arm = Vec::new();

            // A comma can be:
            //  - After the arm. In this case we always want to insert a comma after the newly
            //    inserted arm.
            //  - Missing after the arm, with no arms after. In this case we want to insert a
            //    comma before the newly inserted arm. It can not be necessary if there arm
            //    body is a block, but we don't bother to check that.
            //  - Missing after the arm with arms after, if the arm body is a block. In this case
            //    we don't want to insert a comma at all.
            let has_comma_after =
                std::iter::successors(match_arm.syntax().last_child_or_token(), |it| {
                    it.prev_sibling_or_token()
                })
                .map(|it| it.kind())
                .skip_while(|it| it.is_trivia())
                .next()
                    == Some(T![,]);
            let has_arms_after = neighbor(&match_arm, Direction::Next).is_some();
            if !has_comma_after && !has_arms_after {
                insert_after_old_arm.push(make::token(T![,]).into());
            }

            let indent = IndentLevel::from_node(match_arm.syntax());
            insert_after_old_arm.push(make::tokens::whitespace(&format!("\n{indent}")).into());

            insert_after_old_arm.push(new_match_arm.syntax().clone().into());

            ted::insert_all_raw(Position::after(match_arm.syntax()), insert_after_old_arm);

            if has_comma_after {
                ted::insert_raw(
                    Position::last_child_of(new_match_arm.syntax()),
                    make::token(T![,]),
                );
            }

            edit.replace(old_parent_range, new_parent.to_string());
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
        X::B => 1i32
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
        X::A $0| X::B => {},
    }
}
"#,
            r#"
#[derive(Debug)]
enum X { A, B }

fn main() {
    let x = X::A;
    match x {
        X::A => {},
        X::B => {},
    }
}
"#,
        );
    }
}
