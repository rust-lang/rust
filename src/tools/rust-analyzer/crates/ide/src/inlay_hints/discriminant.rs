//! Implementation of "enum variant discriminant" inlay hints:
//! ```no_run
//! enum Foo {
//!    Bar/* = 0*/,
//! }
//! ```
use ide_db::{base_db::FileId, famous_defs::FamousDefs};
use syntax::ast::{self, AstNode, HasName};

use crate::{DiscriminantHints, InlayHint, InlayHintsConfig, InlayKind, InlayTooltip};

pub(super) fn hints(
    acc: &mut Vec<InlayHint>,
    FamousDefs(sema, _): &FamousDefs<'_, '_>,
    config: &InlayHintsConfig,
    _: FileId,
    variant: &ast::Variant,
) -> Option<()> {
    let field_list = match config.discriminant_hints {
        DiscriminantHints::Always => variant.field_list(),
        DiscriminantHints::Fieldless => match variant.field_list() {
            Some(_) => return None,
            None => None,
        },
        DiscriminantHints::Never => return None,
    };

    if variant.eq_token().is_some() {
        return None;
    }

    let name = variant.name()?;

    let descended = sema.descend_node_into_attributes(variant.clone()).pop();
    let desc_pat = descended.as_ref().unwrap_or(variant);
    let v = sema.to_def(desc_pat)?;
    let d = v.eval(sema.db);

    acc.push(InlayHint {
        range: match field_list {
            Some(field_list) => name.syntax().text_range().cover(field_list.syntax().text_range()),
            None => name.syntax().text_range(),
        },
        kind: InlayKind::DiscriminantHint,
        label: match &d {
            Ok(v) => format!("{}", v).into(),
            Err(_) => "?".into(),
        },
        tooltip: Some(InlayTooltip::String(match &d {
            Ok(_) => "enum variant discriminant".into(),
            Err(e) => format!("{e:?}").into(),
        })),
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
  //^^^^^^^0
    Variant1,
  //^^^^^^^^1
    Variant2,
  //^^^^^^^^2
    Variant5 = 5,
    Variant6,
  //^^^^^^^^6
}
"#,
        );
    }

    #[test]
    fn datacarrying_mixed() {
        check_discriminants(
            r#"
enum Enum {
    Variant(),
  //^^^^^^^^^0
    Variant1,
  //^^^^^^^^1
    Variant2 {},
  //^^^^^^^^^^^2
    Variant3,
  //^^^^^^^^3
    Variant5 = 5,
    Variant6,
  //^^^^^^^^6
}
"#,
        );
    }

    #[test]
    fn datacarrying_mixed_fieldless_set() {
        check_discriminants_fieldless(
            r#"
enum Enum {
    Variant(),
    Variant1,
  //^^^^^^^^1
    Variant2 {},
    Variant3,
  //^^^^^^^^3
    Variant5 = 5,
    Variant6,
  //^^^^^^^^6
}
"#,
        );
    }
}
