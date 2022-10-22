use crate::assist_context::{AssistContext, Assists};
use crate::utils::{replace_arith, ArithKind};

// Assist: replace_arith_with_saturating
//
// Replaces arithmetic on integers with the `saturating_*` equivalent.
//
// ```
// fn main() {
//   let x = 1 $0+ 2;
// }
// ```
// ->
// ```
// fn main() {
//   let x = 1.saturating_add(2);
// }
// ```
pub(crate) fn replace_arith_with_saturating(
    acc: &mut Assists,
    ctx: &AssistContext<'_>,
) -> Option<()> {
    replace_arith(acc, ctx, ArithKind::Saturating)
}

#[cfg(test)]
mod tests {
    use crate::tests::check_assist;

    use super::*;

    #[test]
    fn replace_arith_with_saturating_add() {
        check_assist(
            replace_arith_with_saturating,
            r#"
fn main() {
    let x = 1 $0+ 2;
}
"#,
            r#"
fn main() {
    let x = 1.saturating_add(2);
}
"#,
        )
    }
}
