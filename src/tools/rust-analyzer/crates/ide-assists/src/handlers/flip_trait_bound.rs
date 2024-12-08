use syntax::{
    algo::non_trivia_sibling,
    ast::{self, AstNode},
    Direction, T,
};

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: flip_trait_bound
//
// Flips two trait bounds.
//
// ```
// fn foo<T: Clone +$0 Copy>() { }
// ```
// ->
// ```
// fn foo<T: Copy + Clone>() { }
// ```
pub(crate) fn flip_trait_bound(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    // We want to replicate the behavior of `flip_binexpr` by only suggesting
    // the assist when the cursor is on a `+`
    let plus = ctx.find_token_syntax_at_offset(T![+])?;

    // Make sure we're in a `TypeBoundList`
    ast::TypeBoundList::cast(plus.parent()?)?;

    let (before, after) = (
        non_trivia_sibling(plus.clone().into(), Direction::Prev)?,
        non_trivia_sibling(plus.clone().into(), Direction::Next)?,
    );

    let target = plus.text_range();
    acc.add(
        AssistId("flip_trait_bound", AssistKind::RefactorRewrite),
        "Flip trait bounds",
        target,
        |edit| {
            edit.replace(before.text_range(), after.to_string());
            edit.replace(after.text_range(), before.to_string());
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::tests::{check_assist, check_assist_not_applicable, check_assist_target};

    #[test]
    fn flip_trait_bound_assist_available() {
        check_assist_target(flip_trait_bound, "struct S<T> where T: A $0+ B + C { }", "+")
    }

    #[test]
    fn flip_trait_bound_not_applicable_for_single_trait_bound() {
        check_assist_not_applicable(flip_trait_bound, "struct S<T> where T: $0A { }")
    }

    #[test]
    fn flip_trait_bound_works_for_dyn() {
        check_assist(flip_trait_bound, "fn f<'a>(x: dyn Copy $0+ 'a)", "fn f<'a>(x: dyn 'a + Copy)")
    }

    #[test]
    fn flip_trait_bound_works_for_struct() {
        check_assist(
            flip_trait_bound,
            "struct S<T> where T: A $0+ B { }",
            "struct S<T> where T: B + A { }",
        )
    }

    #[test]
    fn flip_trait_bound_works_for_trait_impl() {
        check_assist(
            flip_trait_bound,
            "impl X for S<T> where T: A +$0 B { }",
            "impl X for S<T> where T: B + A { }",
        )
    }

    #[test]
    fn flip_trait_bound_works_for_fn() {
        check_assist(flip_trait_bound, "fn f<T: A $0+ B>(t: T) { }", "fn f<T: B + A>(t: T) { }")
    }

    #[test]
    fn flip_trait_bound_works_for_fn_where_clause() {
        check_assist(
            flip_trait_bound,
            "fn f<T>(t: T) where T: A +$0 B { }",
            "fn f<T>(t: T) where T: B + A { }",
        )
    }

    #[test]
    fn flip_trait_bound_works_for_lifetime() {
        check_assist(
            flip_trait_bound,
            "fn f<T>(t: T) where T: A $0+ 'static { }",
            "fn f<T>(t: T) where T: 'static + A { }",
        )
    }

    #[test]
    fn flip_trait_bound_works_for_complex_bounds() {
        check_assist(
            flip_trait_bound,
            "struct S<T> where T: A<T> $0+ b_mod::B<T> + C<T> { }",
            "struct S<T> where T: b_mod::B<T> + A<T> + C<T> { }",
        )
    }

    #[test]
    fn flip_trait_bound_works_for_long_bounds() {
        check_assist(
            flip_trait_bound,
            "struct S<T> where T: A + B + C + D + E + F +$0 G + H + I + J { }",
            "struct S<T> where T: A + B + C + D + E + G + F + H + I + J { }",
        )
    }
}
