//! Implementation of "closure return type" inlay hints.
//!
//! Tests live in [`bind_pat`][super::bind_pat] module.
use ide_db::{base_db::FileId, famous_defs::FamousDefs};
use syntax::ast::{self, AstNode};
use text_edit::{TextRange, TextSize};

use crate::{InlayHint, InlayHintLabel, InlayHintPosition, InlayHintsConfig, InlayKind};

pub(super) fn hints(
    acc: &mut Vec<InlayHint>,
    FamousDefs(sema, _): &FamousDefs<'_, '_>,
    config: &InlayHintsConfig,
    _file_id: FileId,
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

    let move_kw_range = match closure.move_token() {
        Some(t) => t.text_range(),
        None => {
            let range = closure.syntax().first_token()?.prev_token()?.text_range();
            let range = TextRange::new(range.end() - TextSize::from(1), range.end());
            acc.push(InlayHint {
                range,
                kind: InlayKind::ClosureCapture,
                label: InlayHintLabel::simple("move", None, None),
                text_edit: None,
                position: InlayHintPosition::After,
                pad_left: false,
                pad_right: false,
            });
            range
        }
    };
    acc.push(InlayHint {
        range: move_kw_range,
        kind: InlayKind::ClosureCapture,
        label: InlayHintLabel::from("("),
        text_edit: None,
        position: InlayHintPosition::After,
        pad_left: false,
        pad_right: false,
    });
    let last = captures.len() - 1;
    for (idx, capture) in captures.into_iter().enumerate() {
        let local = capture.local();
        let source = local.primary_source(sema.db);

        // force cache the source file, otherwise sema lookup will potentially panic
        _ = sema.parse_or_expand(source.file());

        acc.push(InlayHint {
            range: move_kw_range,
            kind: InlayKind::ClosureCapture,
            label: InlayHintLabel::simple(
                format!(
                    "{}{}",
                    match capture.kind() {
                        hir::CaptureKind::SharedRef => "&",
                        hir::CaptureKind::UniqueSharedRef => "&unique ",
                        hir::CaptureKind::MutableRef => "&mut ",
                        hir::CaptureKind::Move => "",
                    },
                    capture.display_place(sema.db)
                ),
                None,
                source.name().and_then(|name| name.syntax().original_file_range_opt(sema.db)),
            ),
            text_edit: None,
            position: InlayHintPosition::After,
            pad_left: false,
            pad_right: false,
        });

        if idx != last {
            acc.push(InlayHint {
                range: move_kw_range,
                kind: InlayKind::ClosureCapture,
                label: InlayHintLabel::simple(", ", None, None),
                text_edit: None,
                position: InlayHintPosition::After,
                pad_left: false,
                pad_right: false,
            });
        }
    }
    acc.push(InlayHint {
        range: move_kw_range,
        kind: InlayKind::ClosureCapture,
        label: InlayHintLabel::from(")"),
        text_edit: None,
        position: InlayHintPosition::After,
        pad_left: false,
        pad_right: true,
    });

    Some(())
}

#[cfg(test)]
mod tests {
    use crate::{
        inlay_hints::tests::{check_with_config, DISABLED_CONFIG},
        InlayHintsConfig,
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
// ^ move
// ^ (
// ^ &foo
// ^ , $
// ^ bar
// ^ , $
// ^ baz
// ^ , $
// ^ qux
// ^ )
        foo;
        bar;
        baz;
        qux;
    };
    || {
// ^ move
// ^ (
// ^ &foo
// ^ , $
// ^ &bar
// ^ , $
// ^ &baz
// ^ , $
// ^ &qux
// ^ )
        &foo;
        &bar;
        &baz;
        &qux;
    };
    || {
// ^ move
// ^ (
// ^ &mut baz
// ^ )
        &mut baz;
    };
    || {
// ^ move
// ^ (
// ^ &mut baz
// ^ , $
// ^ &mut *qux
// ^ )
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
//  ^^^^ (
//  ^^^^ foo
//  ^^^^ )
        foo;
    };
}
"#,
        );
    }
}
