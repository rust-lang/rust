//! Implementation of "closure captures" inlay hints.
//!
//! Tests live in [`bind_pat`][super::bind_pat] module.
use either::Either;
use ide_db::famous_defs::FamousDefs;
use span::Edition;
use stdx::{TupleExt, never};
use syntax::ast::{self, AstNode};

use crate::{
    InlayHint, InlayHintLabel, InlayHintLabelPart, InlayHintPosition, InlayHintsConfig, InlayKind,
};

pub(super) fn hints(
    acc: &mut Vec<InlayHint>,
    FamousDefs(sema, _): &FamousDefs<'_, '_>,
    config: &InlayHintsConfig<'_>,
    expr: Either<ast::ClosureExpr, ast::BlockExpr>,
    edition: Edition,
) -> Option<()> {
    if !config.closure_capture_hints {
        return None;
    }

    let (expr, move_token, capture_anchor) = match expr {
        Either::Left(closure) => {
            let move_token = closure.move_token();
            let capture_anchor = closure.param_list()?.pipe_token()?;
            (closure.into(), move_token, capture_anchor)
        }
        Either::Right(block) => {
            let modifier = block.modifier()?;
            match modifier {
                ast::BlockModifier::Async(_)
                | ast::BlockModifier::Gen(_)
                | ast::BlockModifier::AsyncGen(_) => (),
                ast::BlockModifier::Unsafe(_)
                | ast::BlockModifier::Try { .. }
                | ast::BlockModifier::Const(_)
                | ast::BlockModifier::Label(_) => return None,
            }
            let move_token = block.move_token();
            let capture_anchor = block.stmt_list()?.l_curly_token()?;
            (block.into(), move_token, capture_anchor)
        }
    };

    let ty = &sema.type_of_expr(&expr)?.original;
    let captures = match ty.as_closure() {
        Some(closure) => closure.captured_items(sema.db),
        None => ty.as_coroutine()?.captured_items(sema.db),
    };

    if captures.is_empty() {
        return None;
    }

    let (range, label, position, pad_right) = match move_token {
        Some(token) => {
            (token.text_range(), InlayHintLabel::default(), InlayHintPosition::After, false)
        }
        None => (
            capture_anchor.text_range(),
            InlayHintLabel::from("move"),
            InlayHintPosition::Before,
            true,
        ),
    };
    let mut hint = InlayHint {
        range,
        kind: InlayKind::ClosureCapture,
        label,
        text_edit: None,
        position,
        pad_left: false,
        pad_right,
        resolve_parent: Some(expr.syntax().text_range()),
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
            capture.display_place_source_code(sema.db, edition)
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
//- minicore: copy, derive, fn


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
    fn all_capture_kinds_async_closure() {
        check_with_config(
            InlayHintsConfig { closure_capture_hints: true, ..DISABLED_CONFIG },
            r#"
//- minicore: copy, derive, fn, future, async_fn

#[derive(Copy, Clone)]
struct Copy;

struct NonCopy;

fn main() {
    let foo = Copy;
    let bar = NonCopy;
    let mut baz = NonCopy;
    let qux = &mut NonCopy;
    async || {
       // ^ move(&foo, bar, baz, qux)
        foo;
        bar;
        baz;
        qux;
    };
    async || {
       // ^ move(&foo, &bar, &baz, &qux)
        &foo;
        &bar;
        &baz;
        &qux;
    };
    async || {
       // ^ move(&mut baz)
        &mut baz;
    };
    async || {
       // ^ move(&mut baz, &mut *qux)
        baz = NonCopy;
        *qux = NonCopy;
    };
}

"#,
        );
    }

    #[test]
    fn all_capture_kinds_async_block() {
        check_with_config(
            InlayHintsConfig { closure_capture_hints: true, ..DISABLED_CONFIG },
            r#"
//- minicore: copy, derive, future

#[derive(Copy, Clone)]
struct Copy;

struct NonCopy;

fn main() {
    let foo = Copy;
    let bar = NonCopy;
    let mut baz = NonCopy;
    let qux = &mut NonCopy;
    async {
       // ^ move(&foo, bar, baz, qux)
        foo;
        bar;
        baz;
        qux;
    };
    async {
       // ^ move(&foo, &bar, &baz, &qux)
        &foo;
        &bar;
        &baz;
        &qux;
    };
    async {
       // ^ move(&mut baz)
        &mut baz;
    };
    async {
       // ^ move(&mut baz, &mut *qux)
        baz = NonCopy;
        *qux = NonCopy;
    };
}
"#,
        );
    }

    #[test]
    fn nested_coroutine_does_not_capture_parent_local() {
        check_with_config(
            InlayHintsConfig { closure_capture_hints: true, ..DISABLED_CONFIG },
            r#"
//- minicore: copy, future
fn main() {
    async {
        let foo = 1;
        async {
           // ^ move(&foo)
            foo;
        }
    };
}
"#,
        );
    }

    #[test]
    fn coroutine_blocks() {
        check_with_config(
            InlayHintsConfig { closure_capture_hints: true, ..DISABLED_CONFIG },
            r#"
//- minicore: copy, future
fn main() {
    let foo = 0;
    gen {
     // ^ move(&foo)
        foo;
        yield ();
    };
    async gen {
           // ^ move(&foo)
        foo;
        yield ();
    };
}
"#,
        );
    }

    #[test]
    fn legacy_coroutine() {
        check_with_config(
            InlayHintsConfig { closure_capture_hints: true, ..DISABLED_CONFIG },
            r#"
//- minicore: copy, coroutine
fn main() {
    let foo = 0;
    let coroutine = #[coroutine] || {
                              // ^ move(&foo)
        foo;
        yield ();
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
        check_with_config(
            InlayHintsConfig { closure_capture_hints: true, ..DISABLED_CONFIG },
            r#"
//- minicore: copy, derive
fn main() {
    let foo = u32;
    async move || {
      //  ^^^^ (foo)
        foo;
    };
}
"#,
        );
        check_with_config(
            InlayHintsConfig { closure_capture_hints: true, ..DISABLED_CONFIG },
            r#"
//- minicore: copy, future
fn main() {
    let foo = 0;
    async move {
      //  ^^^^ (foo)
        foo;
    };
}
"#,
        );
    }
}
