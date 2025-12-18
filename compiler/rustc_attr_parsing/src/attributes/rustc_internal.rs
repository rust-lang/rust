use rustc_ast::{LitIntType, LitKind, MetaItemLit};

use super::prelude::*;
use super::util::parse_single_integer;
use crate::session_diagnostics::RustcScalableVectorCountOutOfRange;

pub(crate) struct RustcMainParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcMainParser {
    const PATH: &'static [Symbol] = &[sym::rustc_main];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Fn)]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcMain;
}

pub(crate) struct RustcNeverReturnsNullPointerParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcNeverReturnsNullPointerParser {
    const PATH: &[Symbol] = &[sym::rustc_never_returns_null_ptr];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Fn),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Method(MethodKind::Trait { body: false })),
        Allow(Target::Method(MethodKind::Trait { body: true })),
        Allow(Target::Method(MethodKind::TraitImpl)),
    ]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcNeverReturnsNullPointer;
}

pub(crate) struct RustcLayoutScalarValidRangeStartParser;

impl<S: Stage> SingleAttributeParser<S> for RustcLayoutScalarValidRangeStartParser {
    const PATH: &'static [Symbol] = &[sym::rustc_layout_scalar_valid_range_start];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepInnermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Struct)]);
    const TEMPLATE: AttributeTemplate = template!(List: &["start"]);

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        parse_single_integer(cx, args)
            .map(|n| AttributeKind::RustcLayoutScalarValidRangeStart(Box::new(n), cx.attr_span))
    }
}

pub(crate) struct RustcLayoutScalarValidRangeEndParser;

impl<S: Stage> SingleAttributeParser<S> for RustcLayoutScalarValidRangeEndParser {
    const PATH: &'static [Symbol] = &[sym::rustc_layout_scalar_valid_range_end];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepInnermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Struct)]);
    const TEMPLATE: AttributeTemplate = template!(List: &["end"]);

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        parse_single_integer(cx, args)
            .map(|n| AttributeKind::RustcLayoutScalarValidRangeEnd(Box::new(n), cx.attr_span))
    }
}

pub(crate) struct RustcLegacyConstGenericsParser;

impl<S: Stage> SingleAttributeParser<S> for RustcLegacyConstGenericsParser {
    const PATH: &[Symbol] = &[sym::rustc_legacy_const_generics];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepInnermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Fn)]);
    const TEMPLATE: AttributeTemplate = template!(List: &["N"]);

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        let ArgParser::List(meta_items) = args else {
            cx.expected_list(cx.attr_span, args);
            return None;
        };

        let mut parsed_indexes = ThinVec::new();
        let mut errored = false;

        for possible_index in meta_items.mixed() {
            if let MetaItemOrLitParser::Lit(MetaItemLit {
                kind: LitKind::Int(index, LitIntType::Unsuffixed),
                ..
            }) = possible_index
            {
                parsed_indexes.push((index.0 as usize, possible_index.span()));
            } else {
                cx.expected_integer_literal(possible_index.span());
                errored = true;
            }
        }
        if errored {
            return None;
        } else if parsed_indexes.is_empty() {
            cx.expected_at_least_one_argument(args.span()?);
            return None;
        }

        Some(AttributeKind::RustcLegacyConstGenerics {
            fn_indexes: parsed_indexes,
            attr_span: cx.attr_span,
        })
    }
}

pub(crate) struct RustcObjectLifetimeDefaultParser;

impl<S: Stage> SingleAttributeParser<S> for RustcObjectLifetimeDefaultParser {
    const PATH: &[rustc_span::Symbol] = &[sym::rustc_object_lifetime_default];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepInnermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Struct)]);
    const TEMPLATE: AttributeTemplate = template!(Word);

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        if let Err(span) = args.no_args() {
            cx.expected_no_args(span);
            return None;
        }

        Some(AttributeKind::RustcObjectLifetimeDefault)
    }
}

pub(crate) struct RustcSimdMonomorphizeLaneLimitParser;

impl<S: Stage> SingleAttributeParser<S> for RustcSimdMonomorphizeLaneLimitParser {
    const PATH: &[Symbol] = &[sym::rustc_simd_monomorphize_lane_limit];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepInnermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Struct)]);
    const TEMPLATE: AttributeTemplate = template!(NameValueStr: "N");

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        let ArgParser::NameValue(nv) = args else {
            cx.expected_name_value(cx.attr_span, None);
            return None;
        };
        Some(AttributeKind::RustcSimdMonomorphizeLaneLimit(cx.parse_limit_int(nv)?))
    }
}

pub(crate) struct RustcScalableVectorParser;

impl<S: Stage> SingleAttributeParser<S> for RustcScalableVectorParser {
    const PATH: &[rustc_span::Symbol] = &[sym::rustc_scalable_vector];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepInnermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Struct)]);
    const TEMPLATE: AttributeTemplate = template!(Word, List: &["count"]);

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        if args.no_args().is_ok() {
            return Some(AttributeKind::RustcScalableVector {
                element_count: None,
                span: cx.attr_span,
            });
        }

        let n = parse_single_integer(cx, args)?;
        let Ok(n) = n.try_into() else {
            cx.emit_err(RustcScalableVectorCountOutOfRange { span: cx.attr_span, n });
            return None;
        };
        Some(AttributeKind::RustcScalableVector { element_count: Some(n), span: cx.attr_span })
    }
}
