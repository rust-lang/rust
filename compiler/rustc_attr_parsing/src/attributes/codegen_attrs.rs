use rustc_attr_data_structures::{AttributeKind, OptimizeAttr};
use rustc_feature::{AttributeTemplate, template};
use rustc_session::parse::feature_err;
use rustc_span::{Span, sym};

use super::{AcceptMapping, AttributeOrder, AttributeParser, OnDuplicate, SingleAttributeParser};
use crate::context::{AcceptContext, FinalizeContext, Stage};
use crate::parser::ArgParser;
use crate::session_diagnostics::NakedFunctionIncompatibleAttribute;

pub(crate) struct OptimizeParser;

impl<S: Stage> SingleAttributeParser<S> for OptimizeParser {
    const PATH: &[rustc_span::Symbol] = &[sym::optimize];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepLast;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::WarnButFutureError;
    const TEMPLATE: AttributeTemplate = template!(List: "size|speed|none");

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        let Some(list) = args.list() else {
            cx.expected_list(cx.attr_span);
            return None;
        };

        let Some(single) = list.single() else {
            cx.expected_single_argument(list.span);
            return None;
        };

        let res = match single.meta_item().and_then(|i| i.path().word().map(|i| i.name)) {
            Some(sym::size) => OptimizeAttr::Size,
            Some(sym::speed) => OptimizeAttr::Speed,
            Some(sym::none) => OptimizeAttr::DoNotOptimize,
            _ => {
                cx.expected_specific_argument(single.span(), vec!["size", "speed", "none"]);
                OptimizeAttr::Default
            }
        };

        Some(AttributeKind::Optimize(res, cx.attr_span))
    }
}

pub(crate) struct ColdParser;

impl<S: Stage> SingleAttributeParser<S> for ColdParser {
    const PATH: &[rustc_span::Symbol] = &[sym::cold];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepLast;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const TEMPLATE: AttributeTemplate = template!(Word);

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        if let Err(span) = args.no_args() {
            cx.expected_no_args(span);
            return None;
        }

        Some(AttributeKind::Cold(cx.attr_span))
    }
}

#[derive(Default)]
pub(crate) struct NakedParser {
    span: Option<Span>,
}

impl<S: Stage> AttributeParser<S> for NakedParser {
    const ATTRIBUTES: AcceptMapping<Self, S> =
        &[(&[sym::naked], template!(Word), |this, cx, args| {
            if let Err(span) = args.no_args() {
                cx.expected_no_args(span);
                return;
            }

            if let Some(earlier) = this.span {
                let span = cx.attr_span;
                cx.warn_unused_duplicate(earlier, span);
            } else {
                this.span = Some(cx.attr_span);
            }
        })];

    fn finalize(self, cx: &FinalizeContext<'_, '_, S>) -> Option<AttributeKind> {
        // FIXME(jdonszelmann): upgrade this list to *parsed* attributes
        // once all of these have parsed forms. That'd make the check much nicer...
        //
        // many attributes don't make sense in combination with #[naked].
        // Notable attributes that are incompatible with `#[naked]` are:
        //
        // * `#[inline]`
        // * `#[track_caller]`
        // * `#[test]`, `#[ignore]`, `#[should_panic]`
        //
        // NOTE: when making changes to this list, check that `error_codes/E0736.md` remains
        // accurate.
        const ALLOW_LIST: &[rustc_span::Symbol] = &[
            // conditional compilation
            sym::cfg_trace,
            sym::cfg_attr_trace,
            // testing (allowed here so better errors can be generated in `rustc_builtin_macros::test`)
            sym::test,
            sym::ignore,
            sym::should_panic,
            sym::bench,
            // diagnostics
            sym::allow,
            sym::warn,
            sym::deny,
            sym::forbid,
            sym::deprecated,
            sym::must_use,
            // abi, linking and FFI
            sym::cold,
            sym::export_name,
            sym::link_section,
            sym::linkage,
            sym::no_mangle,
            sym::instruction_set,
            sym::repr,
            sym::rustc_std_internal_symbol,
            sym::align,
            // obviously compatible with self
            sym::naked,
            // documentation
            sym::doc,
        ];

        let span = self.span?;

        // only if we found a naked attribute do we do the somewhat expensive check
        'outer: for other_attr in cx.all_attrs {
            for allowed_attr in ALLOW_LIST {
                if other_attr.segments().next().is_some_and(|i| cx.tools.contains(&i.name)) {
                    // effectively skips the error message  being emitted below
                    // if it's a tool attribute
                    continue 'outer;
                }
                if other_attr.word_is(*allowed_attr) {
                    // effectively skips the error message  being emitted below
                    // if its an allowed attribute
                    continue 'outer;
                }

                if other_attr.word_is(sym::target_feature) {
                    if !cx.features().naked_functions_target_feature() {
                        feature_err(
                            &cx.sess(),
                            sym::naked_functions_target_feature,
                            other_attr.span(),
                            "`#[target_feature(/* ... */)]` is currently unstable on `#[naked]` functions",
                        ).emit();
                    }

                    continue 'outer;
                }
            }

            cx.emit_err(NakedFunctionIncompatibleAttribute {
                span: other_attr.span(),
                naked_span: span,
                attr: other_attr.get_attribute_path().to_string(),
            });
        }

        Some(AttributeKind::Naked(span))
    }
}

pub(crate) struct NoMangleParser;

impl<S: Stage> SingleAttributeParser<S> for NoMangleParser {
    const PATH: &[rustc_span::Symbol] = &[sym::no_mangle];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepLast;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const TEMPLATE: AttributeTemplate = template!(Word);

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        if let Err(span) = args.no_args() {
            cx.expected_no_args(span);
            return None;
        }

        Some(AttributeKind::NoMangle(cx.attr_span))
    }
}
