use std::iter;

use ra_syntax::{
    ast::{
        self,
        edit::{AstNodeEdit, IndentLevel},
        make,
    },
    AstNode,
};

use crate::{
    utils::{render_snippet, Cursor, TryEnum},
    AssistContext, AssistId, AssistKind, Assists,
};

// Assist: replace_unwrap_with_match
//
// Replaces `unwrap` a `match` expression. Works for Result and Option.
//
// ```
// enum Result<T, E> { Ok(T), Err(E) }
// fn main() {
//     let x: Result<i32, i32> = Result::Ok(92);
//     let y = x.<|>unwrap();
// }
// ```
// ->
// ```
// enum Result<T, E> { Ok(T), Err(E) }
// fn main() {
//     let x: Result<i32, i32> = Result::Ok(92);
//     let y = match x {
//         Ok(a) => a,
//         $0_ => unreachable!(),
//     };
// }
// ```
pub(crate) fn replace_unwrap_with_match(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let method_call: ast::MethodCallExpr = ctx.find_node_at_offset()?;
    let name = method_call.name_ref()?;
    if name.text() != "unwrap" {
        return None;
    }
    let caller = method_call.expr()?;
    let ty = ctx.sema.type_of_expr(&caller)?;
    let happy_variant = TryEnum::from_ty(&ctx.sema, &ty)?.happy_case();
    let target = method_call.syntax().text_range();
    acc.add(
        AssistId("replace_unwrap_with_match", AssistKind::RefactorRewrite),
        "Replace unwrap with match",
        target,
        |builder| {
            let ok_path = make::path_unqualified(make::path_segment(make::name_ref(happy_variant)));
            let it = make::bind_pat(make::name("a")).into();
            let ok_tuple = make::tuple_struct_pat(ok_path, iter::once(it)).into();

            let bind_path = make::path_unqualified(make::path_segment(make::name_ref("a")));
            let ok_arm = make::match_arm(iter::once(ok_tuple), make::expr_path(bind_path));

            let unreachable_call = make::expr_unreachable();
            let err_arm =
                make::match_arm(iter::once(make::placeholder_pat().into()), unreachable_call);

            let match_arm_list = make::match_arm_list(vec![ok_arm, err_arm]);
            let match_expr = make::expr_match(caller.clone(), match_arm_list)
                .indent(IndentLevel::from_node(method_call.syntax()));

            let range = method_call.syntax().text_range();
            match ctx.config.snippet_cap {
                Some(cap) => {
                    let err_arm = match_expr
                        .syntax()
                        .descendants()
                        .filter_map(ast::MatchArm::cast)
                        .last()
                        .unwrap();
                    let snippet =
                        render_snippet(cap, match_expr.syntax(), Cursor::Before(err_arm.syntax()));
                    builder.replace_snippet(cap, range, snippet)
                }
                None => builder.replace(range, match_expr.to_string()),
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_target};

    use super::*;

    #[test]
    fn test_replace_result_unwrap_with_match() {
        check_assist(
            replace_unwrap_with_match,
            r"
enum Result<T, E> { Ok(T), Err(E) }
fn i<T>(a: T) -> T { a }
fn main() {
    let x: Result<i32, i32> = Result::Ok(92);
    let y = i(x).<|>unwrap();
}
            ",
            r"
enum Result<T, E> { Ok(T), Err(E) }
fn i<T>(a: T) -> T { a }
fn main() {
    let x: Result<i32, i32> = Result::Ok(92);
    let y = match i(x) {
        Ok(a) => a,
        $0_ => unreachable!(),
    };
}
            ",
        )
    }

    #[test]
    fn test_replace_option_unwrap_with_match() {
        check_assist(
            replace_unwrap_with_match,
            r"
enum Option<T> { Some(T), None }
fn i<T>(a: T) -> T { a }
fn main() {
    let x = Option::Some(92);
    let y = i(x).<|>unwrap();
}
            ",
            r"
enum Option<T> { Some(T), None }
fn i<T>(a: T) -> T { a }
fn main() {
    let x = Option::Some(92);
    let y = match i(x) {
        Some(a) => a,
        $0_ => unreachable!(),
    };
}
            ",
        );
    }

    #[test]
    fn test_replace_result_unwrap_with_match_chaining() {
        check_assist(
            replace_unwrap_with_match,
            r"
enum Result<T, E> { Ok(T), Err(E) }
fn i<T>(a: T) -> T { a }
fn main() {
    let x: Result<i32, i32> = Result::Ok(92);
    let y = i(x).<|>unwrap().count_zeroes();
}
            ",
            r"
enum Result<T, E> { Ok(T), Err(E) }
fn i<T>(a: T) -> T { a }
fn main() {
    let x: Result<i32, i32> = Result::Ok(92);
    let y = match i(x) {
        Ok(a) => a,
        $0_ => unreachable!(),
    }.count_zeroes();
}
            ",
        )
    }

    #[test]
    fn replace_unwrap_with_match_target() {
        check_assist_target(
            replace_unwrap_with_match,
            r"
enum Option<T> { Some(T), None }
fn i<T>(a: T) -> T { a }
fn main() {
    let x = Option::Some(92);
    let y = i(x).<|>unwrap();
}
            ",
            r"i(x).unwrap()",
        );
    }
}
