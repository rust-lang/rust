//! Implementation of "range exclusive" inlay hints:
//! ```ignore
//! for i in 0../* < */10 {}
//! if let ../* < */100 = 50 {}
//! ```
use ide_db::famous_defs::FamousDefs;
use syntax::{SyntaxToken, T, ast};

use crate::{InlayHint, InlayHintsConfig};

pub(super) fn hints(
    acc: &mut Vec<InlayHint>,
    FamousDefs(_sema, _): &FamousDefs<'_, '_>,
    config: &InlayHintsConfig,
    range: impl ast::RangeItem,
) -> Option<()> {
    (config.range_exclusive_hints && range.end().is_some())
        .then(|| {
            range.op_token().filter(|token| token.kind() == T![..]).map(|token| {
                acc.push(inlay_hint(token));
            })
        })
        .flatten()
}

fn inlay_hint(token: SyntaxToken) -> InlayHint {
    InlayHint {
        range: token.text_range(),
        position: crate::InlayHintPosition::After,
        pad_left: false,
        pad_right: false,
        kind: crate::InlayKind::RangeExclusive,
        label: crate::InlayHintLabel::from("<"),
        text_edit: None,
        resolve_parent: None,
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        InlayHintsConfig,
        inlay_hints::tests::{DISABLED_CONFIG, check_with_config},
    };

    #[test]
    fn range_exclusive_expression_bounded_above_hints() {
        check_with_config(
            InlayHintsConfig { range_exclusive_hints: true, ..DISABLED_CONFIG },
            r#"
fn main() {
    let a = 0..10;
           //^^<
    let b = ..100;
          //^^<
    let c = (2 - 1)..(7 * 8)
                 //^^<
}"#,
        );
    }

    #[test]
    fn range_exclusive_expression_unbounded_above_no_hints() {
        check_with_config(
            InlayHintsConfig { range_exclusive_hints: true, ..DISABLED_CONFIG },
            r#"
fn main() {
    let a = 0..;
    let b = ..;
}"#,
        );
    }

    #[test]
    fn range_inclusive_expression_no_hints() {
        check_with_config(
            InlayHintsConfig { range_exclusive_hints: true, ..DISABLED_CONFIG },
            r#"
fn main() {
    let a = 0..=10;
    let b = ..=100;
}"#,
        );
    }

    #[test]
    fn range_exclusive_pattern_bounded_above_hints() {
        check_with_config(
            InlayHintsConfig { range_exclusive_hints: true, ..DISABLED_CONFIG },
            r#"
fn main() {
    if let 0..10 = 0 {}
          //^^<
    if let ..100 = 0 {}
         //^^<
}"#,
        );
    }

    #[test]
    fn range_exclusive_pattern_unbounded_above_no_hints() {
        check_with_config(
            InlayHintsConfig { range_exclusive_hints: true, ..DISABLED_CONFIG },
            r#"
fn main() {
    if let 0.. = 0 {}
    if let .. = 0 {}
}"#,
        );
    }

    #[test]
    fn range_inclusive_pattern_no_hints() {
        check_with_config(
            InlayHintsConfig { range_exclusive_hints: true, ..DISABLED_CONFIG },
            r#"
fn main() {
    if let 0..=10 = 0 {}
    if let ..=100 = 0 {}
}"#,
        );
    }
}
