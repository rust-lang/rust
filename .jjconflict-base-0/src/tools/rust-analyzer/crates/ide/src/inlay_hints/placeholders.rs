//! Implementation of type placeholder inlay hints:
//! ```no_run
//! let a = Vec<_> = vec![4];
//!           //^ = i32
//! ```

use hir::DisplayTarget;
use ide_db::famous_defs::FamousDefs;
use syntax::{
    AstNode,
    ast::{InferType, Type},
};

use crate::{InlayHint, InlayHintPosition, InlayHintsConfig, InlayKind, inlay_hints::label_of_ty};

pub(super) fn type_hints(
    acc: &mut Vec<InlayHint>,
    famous_defs @ FamousDefs(sema, _): &FamousDefs<'_, '_>,
    config: &InlayHintsConfig<'_>,
    display_target: DisplayTarget,
    placeholder: InferType,
) -> Option<()> {
    if !config.type_hints || config.hide_inferred_type_hints {
        return None;
    }

    let syntax = placeholder.syntax();
    let range = syntax.text_range();

    let ty = sema.resolve_type(&Type::InferType(placeholder))?;

    let mut label = label_of_ty(famous_defs, config, &ty, display_target)?;
    label.prepend_str("= ");

    acc.push(InlayHint {
        range,
        kind: InlayKind::Type,
        label,
        text_edit: None,
        position: InlayHintPosition::After,
        pad_left: true,
        pad_right: false,
        resolve_parent: None,
    });
    Some(())
}

#[cfg(test)]
mod tests {
    use crate::{
        InlayHintsConfig,
        inlay_hints::tests::{DISABLED_CONFIG, check_with_config},
    };

    #[track_caller]
    fn check_type_infer(#[rust_analyzer::rust_fixture] ra_fixture: &str) {
        check_with_config(InlayHintsConfig { type_hints: true, ..DISABLED_CONFIG }, ra_fixture);
    }

    #[test]
    fn inferred_types() {
        check_type_infer(
            r#"
struct S<T>(T);

fn foo() {
    let t: (_, _, [_; _]) = (1_u32, S(2), [false] as _);
          //^ = u32
             //^ = S<i32>
                 //^ = bool
                                                   //^ = [bool; 1]
}
"#,
        );
    }

    #[test]
    fn hide_inferred_types() {
        check_with_config(
            InlayHintsConfig {
                type_hints: true,
                hide_inferred_type_hints: true,
                ..DISABLED_CONFIG
            },
            r#"
struct S<T>(T);

fn foo() {
    let t: (_, _, [_; _]) = (1_u32, S(2), [false] as _);
}
        "#,
        );
    }
}
