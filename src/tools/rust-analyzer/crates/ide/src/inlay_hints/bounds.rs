//! Implementation of trait bound hints.
//!
//! Currently this renders the implied `Sized` bound.
use ide_db::{FileRange, famous_defs::FamousDefs};

use syntax::ast::{self, AstNode, HasTypeBounds};

use crate::{
    InlayHint, InlayHintLabel, InlayHintLabelPart, InlayHintPosition, InlayHintsConfig, InlayKind,
    TryToNav,
};

pub(super) fn hints(
    acc: &mut Vec<InlayHint>,
    famous_defs @ FamousDefs(sema, _): &FamousDefs<'_, '_>,
    config: &InlayHintsConfig,
    params: ast::GenericParamList,
) -> Option<()> {
    if !config.sized_bound {
        return None;
    }

    let sized_trait = famous_defs.core_marker_Sized();

    for param in params.type_or_const_params() {
        match param {
            ast::TypeOrConstParam::Type(type_param) => {
                let c = type_param.colon_token().map(|it| it.text_range());
                let has_bounds =
                    type_param.type_bound_list().is_some_and(|it| it.bounds().next().is_some());
                acc.push(InlayHint {
                    range: c.unwrap_or_else(|| type_param.syntax().text_range()),
                    kind: InlayKind::Type,
                    label: {
                        let mut hint = InlayHintLabel::default();
                        if c.is_none() {
                            hint.parts.push(InlayHintLabelPart {
                                text: ": ".to_owned(),
                                linked_location: None,
                                tooltip: None,
                            });
                        }
                        hint.parts.push(InlayHintLabelPart {
                            text: "Sized".to_owned(),
                            linked_location: sized_trait.and_then(|it| {
                                config.lazy_location_opt(|| {
                                    it.try_to_nav(sema.db).map(|it| {
                                        let n = it.call_site();
                                        FileRange {
                                            file_id: n.file_id,
                                            range: n.focus_or_full_range(),
                                        }
                                    })
                                })
                            }),
                            tooltip: None,
                        });
                        if has_bounds {
                            hint.parts.push(InlayHintLabelPart {
                                text: " +".to_owned(),
                                linked_location: None,
                                tooltip: None,
                            });
                        }
                        hint
                    },
                    text_edit: None,
                    position: InlayHintPosition::After,
                    pad_left: c.is_some(),
                    pad_right: has_bounds,
                    resolve_parent: Some(params.syntax().text_range()),
                });
            }
            ast::TypeOrConstParam::Const(_) => (),
        }
    }

    Some(())
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use crate::inlay_hints::InlayHintsConfig;

    use crate::inlay_hints::tests::{DISABLED_CONFIG, check_expect, check_with_config};

    #[track_caller]
    fn check(#[rust_analyzer::rust_fixture] ra_fixture: &str) {
        check_with_config(InlayHintsConfig { sized_bound: true, ..DISABLED_CONFIG }, ra_fixture);
    }

    #[test]
    fn smoke() {
        check(
            r#"
fn foo<T>() {}
    // ^ : Sized
"#,
        );
    }

    #[test]
    fn with_colon() {
        check(
            r#"
fn foo<T:>() {}
     // ^ Sized
"#,
        );
    }

    #[test]
    fn with_colon_and_bounds() {
        check(
            r#"
fn foo<T: 'static>() {}
     // ^ Sized +
"#,
        );
    }

    #[test]
    fn location_works() {
        check_expect(
            InlayHintsConfig { sized_bound: true, ..DISABLED_CONFIG },
            r#"
//- minicore: sized
fn foo<T>() {}
"#,
            expect![[r#"
                [
                    (
                        7..8,
                        [
                            ": ",
                            InlayHintLabelPart {
                                text: "Sized",
                                linked_location: Some(
                                    Computed(
                                        FileRangeWrapper {
                                            file_id: FileId(
                                                1,
                                            ),
                                            range: 135..140,
                                        },
                                    ),
                                ),
                                tooltip: "",
                            },
                        ],
                    ),
                ]
            "#]],
        );
    }
}
