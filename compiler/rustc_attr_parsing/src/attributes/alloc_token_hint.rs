use rustc_ast::ast::{LitKind, MetaItemLit};
use rustc_feature::AttributeStability;

use super::prelude::*;
use crate::session_diagnostics::{AllocTokenHintExpectedBool, AllocTokenHintMissingArgs};

pub(crate) struct AllocTokenHintParser;

impl SingleAttributeParser for AllocTokenHintParser {
    const PATH: &[Symbol] = &[sym::alloc_token_hint];
    const ALLOWED_TARGETS: AllowedTargets<'_> = AllowedTargets::AllowListWarnRest(&[
        Allow(Target::Struct),
        Allow(Target::Enum),
        Allow(Target::Union),
    ]);
    const TEMPLATE: AttributeTemplate = template!(
        List: &[
            r#"contains_pointers = <bool>"#,
            r#"type_name = "...""#,
            r#"contains_pointers = <bool>, type_name = "...""#,
        ]
    );
    const STABILITY: AttributeStability = unstable!(alloc_token_hint);

    fn convert(cx: &mut AcceptContext<'_, '_>, args: &ArgParser) -> Option<AttributeKind> {
        let list = cx.expect_list(args, cx.attr_span)?;

        let mut contains_pointers = None;
        let mut type_name = None;

        for param in list.mixed() {
            let Some(param) = param.meta_item() else {
                cx.adcx().expected_not_literal(param.span());
                return None;
            };

            let ident_name = param.path().word_sym();

            match ident_name {
                Some(name @ sym::contains_pointers) => {
                    if contains_pointers.is_some() {
                        cx.adcx().duplicate_key(param.span(), name);
                        return None;
                    }
                    let nv = cx.expect_name_value(param.args(), param.span(), Some(name))?;
                    let MetaItemLit { kind: LitKind::Bool(value), .. } = nv.value_as_lit() else {
                        cx.emit_err(AllocTokenHintExpectedBool { span: nv.value_span });
                        return None;
                    };
                    contains_pointers = Some(*value);
                }
                Some(name @ sym::type_name) => {
                    if type_name.is_some() {
                        cx.adcx().duplicate_key(param.span(), name);
                        return None;
                    }
                    let nv = cx.expect_name_value(param.args(), param.span(), Some(name))?;
                    type_name = Some(cx.expect_string_literal(nv)?);
                }
                _ => {
                    cx.adcx().expected_specific_argument(
                        param.span(),
                        &[sym::contains_pointers, sym::type_name],
                    );
                    return None;
                }
            }
        }

        if contains_pointers.is_none() && type_name.is_none() {
            cx.emit_err(AllocTokenHintMissingArgs { span: cx.attr_span });
            return None;
        }

        Some(AttributeKind::AllocTokenHint { contains_pointers, type_name })
    }
}
