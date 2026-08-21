use super::prelude::*;

// TODO: Make a discussion in writing how similar it is to the Clang's done loopbound work in the src/llvm-project/clang/test/CodeGen/Patmos/loopbounds.c et. al.

pub(crate) struct LoopBoundParser;

impl<S: Stage> SingleAttributeParser<S> for LoopBoundParser {
    const PATH: &[Symbol] = &[sym::loop_bound];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Expression)]);
    const TEMPLATE: AttributeTemplate = template!(
        List: &[r#"min = "value", max = "value""#],
        "Loop bound attribute for passing bounds to LLVM"
    );
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepInnermost;

    // TODO: Need to do better explaination of this.
    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        let ArgParser::List(list) = args else {
            cx.expected_list(cx.attr_span, args);
            return None;
        };

        let mut min_value: Option<u64> = None;
        let mut max_value: Option<u64> = None;

        for param in list.mixed() {
            let Some(item) = param.meta_item() else {
                cx.unexpected_literal(param.span());
                return None;
            };

            let Some(name) = item.path().word().map(|ident| ident.name) else {
                cx.emit_err(crate::session_diagnostics::InvalidAlignmentValue {
                    span: item.span(),
                    error_part: "expected a simple identifier",
                });
                return None;
            };

            let Some(nv) = item.args().name_value() else {
                cx.expected_name_value(item.span(), None);
                return None;
            };

            let value_lit = nv.value_as_lit();

            if name.as_str() == "min" {
                if let rustc_ast::LitKind::Int(min_val, _) = value_lit.kind {
                    match min_val.0.try_into() {
                        Ok(val) => {
                            if min_value.is_some() {
                                cx.duplicate_key(nv.value_span, name);
                                return None;
                            }
                            min_value = Some(val);
                        }
                        Err(_) => {
                            cx.emit_err(crate::session_diagnostics::InvalidAlignmentValue {
                                span: nv.value_span,
                                error_part: "loop bound min value is too large for u64",
                            });
                            return None;
                        }
                    }
                } else {
                    cx.expected_integer_literal(nv.value_span);
                    return None;
                }
            } else if name.as_str() == "max" {
                if let rustc_ast::LitKind::Int(max_val, _) = value_lit.kind {
                    match max_val.0.try_into() {
                        Ok(val) => {
                            if max_value.is_some() {
                                cx.duplicate_key(nv.value_span, name);
                                return None;
                            }
                            max_value = Some(val);
                        }
                        Err(_) => {
                            cx.emit_err(crate::session_diagnostics::InvalidAlignmentValue {
                                span: nv.value_span,
                                error_part: "loop bound max value is too large for u64",
                            });
                            return None;
                        }
                    }
                } else {
                    cx.expected_integer_literal(nv.value_span);
                    return None;
                }
            } else {
                cx.emit_err(crate::session_diagnostics::InvalidAlignmentValue {
                    span: item.span(),
                    error_part: "unexpected name in loop_bound attribute",
                });
                return None;
            }
        }

        let min_value = match min_value {
            Some(val) => val,
            None => {
                cx.expected_name_value(cx.attr_span, None);
                return None;
            }
        };

        let max_value = match max_value {
            Some(val) => val,
            None => {
                cx.expected_name_value(cx.attr_span, None);
                return None;
            }
        };

        // Validate that min <= max
        if min_value > max_value {
            cx.emit_err(crate::session_diagnostics::LoopBoundInvalidRange {
                span: cx.attr_span,
                min: min_value,
                max: max_value,
            });
            return None;
        }

        Some(AttributeKind::LoopBound {
            min: min_value,
            max: max_value,
            span: cx.attr_span,
        })
    }
}
