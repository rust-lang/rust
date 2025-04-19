//! Implementation of "closure captures" inlay hints.
//!
//! Tests live in [`bind_pat`][super::bind_pat] module.
use ide_db::famous_defs::FamousDefs;
use ide_db::text_edit::{TextRange, TextSize};
use stdx::{TupleExt, never};
use syntax::ast::{self, AstNode};

use crate::{
    InlayHint, InlayHintLabel, InlayHintLabelPart, InlayHintPosition, InlayHintsConfig, InlayKind,
};

pub(super) fn hints(
    acc: &mut Vec<InlayHint>,
    FamousDefs(sema, _): &FamousDefs<'_, '_>,
    config: &InlayHintsConfig,
    closure: ast::ClosureExpr,
) -> Option<()> {
    if !config.closure_capture_hints {
        return None;
    }
    let ty = &sema.type_of_expr(&closure.clone().into())?.original;
    let c = ty.as_closure()?;
    let captures = c.captured_items(sema.db);

    if captures.is_empty() {
        return None;
    }

    let (range, label) = match closure.move_token() {
        Some(t) => (t.text_range(), InlayHintLabel::default()),
        None => {
            let prev_token = closure.syntax().first_token()?.prev_token()?.text_range();
            (
                TextRange::new(prev_token.end() - TextSize::from(1), prev_token.end()),
                InlayHintLabel::from("move"),
            )
        }
    };
    let mut hint = InlayHint {
        range,
        kind: InlayKind::ClosureCapture,
        label,
        text_edit: None,
        position: InlayHintPosition::After,
        pad_left: false,
        pad_right: true,
        resolve_parent: Some(closure.syntax().text_range()),
    };
    hint.label.append_str("(");
    let last = captures.len() - 1;
    for (idx, capture) in captures.into_iter().enumerate() {
        let local = capture.local();

        let label = format!(
            "{}{}",
            match capture.kind() {
                hir::CaptureKind::SharedRef => "&",
                hir::CaptureKind::UniqueSharedRef => "&unique ",
                hir::CaptureKind::MutableRef => "&mut ",
                hir::CaptureKind::Move => "",
            },
            capture.display_place(sema.db)
        );
        if never!(label.is_empty()) {
            continue;
        }
        hint.label.append_part(InlayHintLabelPart {
            text: label,
            linked_location: config.lazy_location_opt(|| {
                let source = local.primary_source(sema.db);

                // force cache the source file, otherwise sema lookup will potentially panic
                _ = sema.parse_or_expand(source.file());
                source.name().and_then(|name| {
                    name.syntax().original_file_range_opt(sema.db).map(TupleExt::head).map(
                        |frange| ide_db::FileRange {
                            file_id: frange.file_id.file_id(sema.db),
                            range: frange.range,
                        },
                    )
                })
            }),
            tooltip: None,
        });

        if idx != last {
            hint.label.append_str(", ");
        }
    }
    hint.label.append_str(")");
    acc.push(hint);
    Some(())
}

#[cfg(test)]
mod tests {
    use crate::{
        InlayHintsConfig,
        inlay_hints::tests::{DISABLED_CONFIG, check_with_config},
    };

    #[test]
    fn all_capture_kinds() {
        check_with_config(
            InlayHintsConfig { closure_capture_hints: true, ..DISABLED_CONFIG },
            r#"
//- minicore: copy, derive


#[derive(Copy, Clone)]
struct Copy;

struct NonCopy;

fn main() {
    let foo = Copy;
    let bar = NonCopy;
    let mut baz = NonCopy;
    let qux = &mut NonCopy;
    || {
// ^ move(&foo, bar, baz, qux)
        foo;
        bar;
        baz;
        qux;
    };
    || {
// ^ move(&foo, &bar, &baz, &qux)
        &foo;
        &bar;
        &baz;
        &qux;
    };
    || {
// ^ move(&mut baz)
        &mut baz;
    };
    || {
// ^ move(&mut baz, &mut *qux)
        baz = NonCopy;
        *qux = NonCopy;
    };
}
"#,
        );
    }

    #[test]
    fn move_token() {
        check_with_config(
            InlayHintsConfig { closure_capture_hints: true, ..DISABLED_CONFIG },
            r#"
//- minicore: copy, derive
fn main() {
    let foo = u32;
    move || {
//  ^^^^ (foo)
        foo;
    };
}
"#,
        );
    }
}
