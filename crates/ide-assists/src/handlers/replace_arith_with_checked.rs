use crate::assist_context::{AssistContext, Assists};
use crate::utils::{replace_arith, ArithKind};

pub(crate) fn replace_arith_with_checked(
    acc: &mut Assists,
    ctx: &AssistContext<'_>,
) -> Option<()> {
    replace_arith(acc, ctx, ArithKind::Checked)
}

#[cfg(test)]
mod tests {
    use crate::tests::check_assist;

    use super::*;

    #[test]
    fn replace_arith_with_saturating_add() {
        check_assist(
            replace_arith_with_checked,
            r#"
fn main() {
    let x = 1 $0+ 2;
}
"#,
            r#"
fn main() {
    let x = 1.checked_add(2);
}
"#,
        )
    }
}
