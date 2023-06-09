//! Implementation of "enum variant discriminant" inlay hints:
//! ```no_run
//! enum Foo {
//!    Bar/* = 0*/,
//! }
//! ```
use hir::Semantics;
use ide_db::{base_db::FileId, famous_defs::FamousDefs, RootDatabase};
use syntax::ast::{self, AstNode, HasName};

use crate::{
    DiscriminantHints, InlayHint, InlayHintLabel, InlayHintPosition, InlayHintsConfig, InlayKind,
    InlayTooltip,
};

pub(super) fn enum_hints(
    acc: &mut Vec<InlayHint>,
    FamousDefs(sema, _): &FamousDefs<'_, '_>,
    config: &InlayHintsConfig,
    _: FileId,
    enum_: ast::Enum,
) -> Option<()> {
    if let DiscriminantHints::Never = config.discriminant_hints {
        return None;
    }

    let def = sema.to_def(&enum_)?;
    let data_carrying = def.is_data_carrying(sema.db);
    if matches!(config.discriminant_hints, DiscriminantHints::Fieldless) && data_carrying {
        return None;
    }
    // data carrying enums without a primitive repr have no stable discriminants
    if data_carrying && def.repr(sema.db).map_or(true, |r| r.int.is_none()) {
        return None;
    }
    for variant in enum_.variant_list()?.variants() {
        variant_hints(acc, sema, &variant);
    }
    Some(())
}

fn variant_hints(
    acc: &mut Vec<InlayHint>,
    sema: &Semantics<'_, RootDatabase>,
    variant: &ast::Variant,
) -> Option<()> {
    if variant.expr().is_some() {
        return None;
    }

    let eq_token = variant.eq_token();
    let name = variant.name()?;

    let descended = sema.descend_node_into_attributes(variant.clone()).pop();
    let desc_pat = descended.as_ref().unwrap_or(variant);
    let v = sema.to_def(desc_pat)?;
    let d = v.eval(sema.db);

    let range = match variant.field_list() {
        Some(field_list) => name.syntax().text_range().cover(field_list.syntax().text_range()),
        None => name.syntax().text_range(),
    };
    let eq_ = if eq_token.is_none() { " =" } else { "" };
    let label = InlayHintLabel::simple(
        match d {
            Ok(x) => {
                if x >= 10 {
                    format!("{eq_} {x} ({x:#X})")
                } else {
                    format!("{eq_} {x}")
                }
            }
            Err(_) => format!("{eq_} ?"),
        },
        Some(InlayTooltip::String(match &d {
            Ok(_) => "enum variant discriminant".into(),
            Err(e) => format!("{e:?}").into(),
        })),
        None,
    );
    acc.push(InlayHint {
        range: match eq_token {
            Some(t) => range.cover(t.text_range()),
            _ => range,
        },
        kind: InlayKind::Discriminant,
        label,
        text_edit: None,
        position: InlayHintPosition::After,
        pad_left: false,
        pad_right: false,
    });

    Some(())
}
#[cfg(test)]
mod tests {
    use crate::inlay_hints::{
        tests::{check_with_config, DISABLED_CONFIG},
        DiscriminantHints, InlayHintsConfig,
    };

    #[track_caller]
    fn check_discriminants(ra_fixture: &str) {
        check_with_config(
            InlayHintsConfig { discriminant_hints: DiscriminantHints::Always, ..DISABLED_CONFIG },
            ra_fixture,
        );
    }

    #[track_caller]
    fn check_discriminants_fieldless(ra_fixture: &str) {
        check_with_config(
            InlayHintsConfig {
                discriminant_hints: DiscriminantHints::Fieldless,
                ..DISABLED_CONFIG
            },
            ra_fixture,
        );
    }

    #[test]
    fn fieldless() {
        check_discriminants(
            r#"
enum Enum {
    Variant,
//  ^^^^^^^ = 0$
    Variant1,
//  ^^^^^^^^ = 1$
    Variant2,
//  ^^^^^^^^ = 2$
    Variant5 = 5,
    Variant6,
//  ^^^^^^^^ = 6$
}
"#,
        );
        check_discriminants_fieldless(
            r#"
enum Enum {
    Variant,
//  ^^^^^^^ = 0
    Variant1,
//  ^^^^^^^^ = 1
    Variant2,
//  ^^^^^^^^ = 2
    Variant5 = 5,
    Variant6,
//  ^^^^^^^^ = 6
}
"#,
        );
    }

    #[test]
    fn datacarrying_mixed() {
        check_discriminants(
            r#"
#[repr(u8)]
enum Enum {
    Variant(),
//  ^^^^^^^^^ = 0
    Variant1,
//  ^^^^^^^^ = 1
    Variant2 {},
//  ^^^^^^^^^^^ = 2
    Variant3,
//  ^^^^^^^^ = 3
    Variant5 = 5,
    Variant6,
//  ^^^^^^^^ = 6
}
"#,
        );
        check_discriminants(
            r#"
enum Enum {
    Variant(),
    Variant1,
    Variant2 {},
    Variant3,
    Variant5,
    Variant6,
}
"#,
        );
    }

    #[test]
    fn datacarrying_mixed_fieldless_set() {
        check_discriminants_fieldless(
            r#"
#[repr(u8)]
enum Enum {
    Variant(),
    Variant1,
    Variant2 {},
    Variant3,
    Variant5,
    Variant6,
}
"#,
        );
    }
}
