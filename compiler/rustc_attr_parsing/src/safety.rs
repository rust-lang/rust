use rustc_ast::Safety;
use rustc_feature::{AttributeSafety, BUILTIN_ATTRIBUTE_MAP};
use rustc_hir::AttrPath;
use rustc_hir::lints::{AttributeLint, AttributeLintKind};
use rustc_session::lint::LintId;
use rustc_session::lint::builtin::UNSAFE_ATTR_OUTSIDE_UNSAFE;
use rustc_span::Span;

use crate::context::Stage;
use crate::{AttributeParser, ShouldEmit};

impl<'sess, S: Stage> AttributeParser<'sess, S> {
    pub fn check_attribute_safety(
        &mut self,
        attr_path: &AttrPath,
        attr_span: Span,
        attr_safety: Safety,
        emit_lint: &mut impl FnMut(AttributeLint<S::Id>),
        target_id: S::Id,
    ) {
        if matches!(self.stage.should_emit(), ShouldEmit::Nothing) {
            return;
        }

        let name = (attr_path.segments.len() == 1).then_some(attr_path.segments[0]);

        // FIXME: We should retrieve this information from the attribute parsers instead of from `BUILTIN_ATTRIBUTE_MAP`
        let builtin_attr_info = name.and_then(|name| BUILTIN_ATTRIBUTE_MAP.get(&name));
        let builtin_attr_safety = builtin_attr_info.map(|x| x.safety);

        match (builtin_attr_safety, attr_safety) {
            // - Unsafe builtin attribute
            // - User wrote `#[unsafe(..)]`, which is permitted on any edition
            (Some(AttributeSafety::Unsafe { .. }), Safety::Unsafe(..)) => {
                // OK
            }

            // - Unsafe builtin attribute
            // - User did not write `#[unsafe(..)]`
            (Some(AttributeSafety::Unsafe { unsafe_since }), Safety::Default) => {
                let path_span = attr_path.span;

                // If the `attr_item`'s span is not from a macro, then just suggest
                // wrapping it in `unsafe(...)`. Otherwise, we suggest putting the
                // `unsafe(`, `)` right after and right before the opening and closing
                // square bracket respectively.
                let diag_span = attr_span;

                // Attributes can be safe in earlier editions, and become unsafe in later ones.
                //
                // Use the span of the attribute's name to determine the edition: the span of the
                // attribute as a whole may be inaccurate if it was emitted by a macro.
                //
                // See https://github.com/rust-lang/rust/issues/142182.
                let emit_error = match unsafe_since {
                    None => true,
                    Some(unsafe_since) => path_span.edition() >= unsafe_since,
                };

                let mut not_from_proc_macro = true;
                if diag_span.from_expansion()
                    && let Ok(mut snippet) = self.sess.source_map().span_to_snippet(diag_span)
                {
                    snippet.retain(|c| !c.is_whitespace());
                    if snippet.contains("!(") || snippet.starts_with("#[") && snippet.ends_with("]")
                    {
                        not_from_proc_macro = false;
                    }
                }

                if emit_error {
                    self.stage.emit_err(
                        self.sess,
                        crate::session_diagnostics::UnsafeAttrOutsideUnsafe {
                            span: path_span,
                            suggestion: not_from_proc_macro.then(|| {
                                crate::session_diagnostics::UnsafeAttrOutsideUnsafeSuggestion {
                                    left: diag_span.shrink_to_lo(),
                                    right: diag_span.shrink_to_hi(),
                                }
                            }),
                        },
                    );
                } else {
                    emit_lint(AttributeLint {
                        lint_id: LintId::of(UNSAFE_ATTR_OUTSIDE_UNSAFE),
                        id: target_id,
                        span: path_span,
                        kind: AttributeLintKind::UnsafeAttrOutsideUnsafe {
                            attribute_name_span: path_span,
                            sugg_spans: not_from_proc_macro
                                .then(|| (diag_span.shrink_to_lo(), diag_span.shrink_to_hi())),
                        },
                    })
                }
            }

            // - Normal builtin attribute
            // - Writing `#[unsafe(..)]` is not permitted on normal builtin attributes
            (None | Some(AttributeSafety::Normal), Safety::Unsafe(unsafe_span)) => {
                self.stage.emit_err(
                    self.sess,
                    crate::session_diagnostics::InvalidAttrUnsafe {
                        span: unsafe_span,
                        name: attr_path.clone(),
                    },
                );
            }

            // - Normal builtin attribute
            // - No explicit `#[unsafe(..)]` written.
            (None | Some(AttributeSafety::Normal), Safety::Default) => {
                // OK
            }

            (
                Some(AttributeSafety::Unsafe { .. } | AttributeSafety::Normal) | None,
                Safety::Safe(..),
            ) => {
                self.sess.dcx().span_delayed_bug(
                    attr_span,
                    "`check_attribute_safety` does not expect `Safety::Safe` on attributes",
                );
            }
        }
    }
}
