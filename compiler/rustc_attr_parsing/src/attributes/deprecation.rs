use rustc_ast::LitKind;
use rustc_hir::attrs::{DeprecatedSince, Deprecation};
use rustc_hir::{RustcVersion, VERSION_PLACEHOLDER};

use super::prelude::*;
use super::util::parse_version;
use crate::session_diagnostics::{
    DeprecatedItemSuggestion, InvalidSince, MissingNote, MissingSince,
};

fn get(
    cx: &mut AcceptContext<'_, '_>,
    name: Symbol,
    param_span: Span,
    arg: &ArgParser,
    item: Option<Symbol>,
) -> Option<Ident> {
    if item.is_some() {
        cx.adcx().duplicate_key(param_span, name);
        return None;
    }
    let v = cx.expect_name_value(arg, param_span, Some(name))?;
    if let Some(value_str) = v.value_as_ident() {
        Some(value_str)
    } else {
        cx.adcx().expected_string_literal(v.value_span, Some(&v.value_as_lit()));
        None
    }
}

pub(crate) struct DeprecatedParser;
impl SingleAttributeParser for DeprecatedParser {
    const PATH: &[Symbol] = &[sym::deprecated];
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowListWarnRest(&[
        Allow(Target::Fn),
        Allow(Target::Mod),
        Allow(Target::Struct),
        Allow(Target::Enum),
        Allow(Target::Union),
        Allow(Target::Const),
        Allow(Target::Static),
        Allow(Target::MacroDef),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Method(MethodKind::Trait { body: false })),
        Allow(Target::Method(MethodKind::Trait { body: true })),
        Allow(Target::TyAlias),
        Allow(Target::Use),
        Allow(Target::ForeignFn),
        Allow(Target::ForeignStatic),
        Allow(Target::ForeignTy),
        Allow(Target::Field),
        Allow(Target::Trait),
        Allow(Target::AssocTy),
        Allow(Target::AssocConst),
        Allow(Target::Variant),
        Allow(Target::Impl { of_trait: false }),
        Allow(Target::Crate),
        Error(Target::WherePredicate),
    ]);
    const TEMPLATE: AttributeTemplate = template!(
        Word,
        List: &[r#"since = "version""#, r#"note = "reason""#, r#"since = "version", note = "reason""#],
        NameValueStr: "reason"
    );

    fn convert(cx: &mut AcceptContext<'_, '_>, args: &ArgParser) -> Option<AttributeKind> {
        let features = cx.features();

        let mut since = None;
        let mut note: Option<Ident> = None;
        let mut suggestion = None;

        let is_rustc = features.staged_api();

        match args {
            ArgParser::NoArgs => {
                // ok
            }
            ArgParser::List(list) => {
                // If the argument list contains a single string literal:
                // check whether it may be a version and suggest since field
                // otherwise, suggest using NameValue syntax
                if let Some(elem) = list.as_single()
                    && let Some(lit) = elem.as_lit()
                    && let LitKind::Str(text, _) = lit.kind
                {
                    let mut adcx = cx.adcx();

                    match parse_since(text, true) {
                        DeprecatedSince::Future | DeprecatedSince::RustcVersion(_) => {
                            adcx.push_suggestion(
                                String::from("try specifying a deprecated since version"),
                                elem.span(),
                                format!("since = {}", lit.kind),
                            );
                        }
                        _ => {
                            if let Some(span) = args.span() {
                                adcx.push_suggestion(
                                    String::from("try using `=` instead"),
                                    span,
                                    format!(" = {}", lit.kind),
                                );
                            }
                        }
                    };

                    adcx.expected_not_literal(elem.span());
                    return None;
                }

                for param in list.mixed() {
                    let Some(param) = param.meta_item() else {
                        cx.adcx().expected_not_literal(param.span());
                        return None;
                    };

                    let ident_name = param.path().word_sym();

                    match ident_name {
                        Some(name @ sym::since) => {
                            since = Some(get(cx, name, param.span(), param.args(), since)?.name);
                        }
                        Some(name @ sym::note) => {
                            note = Some(get(
                                cx,
                                name,
                                param.span(),
                                param.args(),
                                note.map(|ident| ident.name),
                            )?);
                        }
                        Some(name @ sym::suggestion) => {
                            if !features.deprecated_suggestion() {
                                cx.emit_err(DeprecatedItemSuggestion {
                                    span: param.span(),
                                    is_nightly: cx.sess().is_nightly_build(),
                                    details: (),
                                });
                            }

                            suggestion =
                                Some(get(cx, name, param.span(), param.args(), suggestion)?.name);
                        }
                        _ => {
                            cx.adcx().expected_specific_argument(
                                param.span(),
                                if features.deprecated_suggestion() {
                                    &[sym::since, sym::note, sym::suggestion]
                                } else {
                                    &[sym::since, sym::note]
                                },
                            );
                            return None;
                        }
                    }
                }
            }
            ArgParser::NameValue(v) => {
                let Some(value) = v.value_as_ident() else {
                    cx.adcx().expected_string_literal(v.value_span, Some(v.value_as_lit()));
                    return None;
                };
                note = Some(value);
            }
        }

        let since = if let Some(since) = since {
            let since = parse_since(since, is_rustc);
            if matches!(since, DeprecatedSince::Err) {
                cx.emit_err(InvalidSince { span: cx.attr_span });
            }
            since
        } else if is_rustc {
            cx.emit_err(MissingSince { span: cx.attr_span });
            DeprecatedSince::Err
        } else {
            DeprecatedSince::Unspecified
        };

        if is_rustc && note.is_none() {
            cx.emit_err(MissingNote { span: cx.attr_span });
            return None;
        }

        Some(AttributeKind::Deprecated {
            deprecation: Deprecation { since, note, suggestion },
            span: cx.attr_span,
        })
    }
}

fn parse_since(since: Symbol, is_rustc: bool) -> DeprecatedSince {
    if since.as_str() == "TBD" {
        DeprecatedSince::Future
    } else if !is_rustc {
        DeprecatedSince::NonStandard(since)
    } else if since.as_str() == VERSION_PLACEHOLDER {
        DeprecatedSince::RustcVersion(RustcVersion::CURRENT)
    } else if let Some(version) = parse_version(since) {
        DeprecatedSince::RustcVersion(version)
    } else {
        DeprecatedSince::Err
    }
}
