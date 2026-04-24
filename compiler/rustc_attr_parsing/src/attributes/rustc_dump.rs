use rustc_hir::attrs::{AttributeKind, RustcDumpLayoutKind};
use rustc_hir::{MethodKind, Target};
use rustc_span::{Span, Symbol, sym};

use super::prelude::*;
use crate::context::Stage;
use crate::target_checking::AllowedTargets;

pub(crate) struct RustcDumpUserArgsParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcDumpUserArgsParser {
    const PATH: &[Symbol] = &[sym::rustc_dump_user_args];
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Fn)]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcDumpUserArgs;
}

pub(crate) struct RustcDumpDefParentsParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcDumpDefParentsParser {
    const PATH: &[Symbol] = &[sym::rustc_dump_def_parents];
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Fn)]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcDumpDefParents;
}

pub(crate) struct RustcDumpDefPathParser;

impl<S: Stage> SingleAttributeParser<S> for RustcDumpDefPathParser {
    const PATH: &[Symbol] = &[sym::rustc_dump_def_path];
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Fn),
        Allow(Target::Method(MethodKind::TraitImpl)),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Method(MethodKind::Trait { body: true })),
        Allow(Target::ForeignFn),
        Allow(Target::ForeignStatic),
        Allow(Target::Impl { of_trait: false }),
    ]);
    const TEMPLATE: AttributeTemplate = template!(Word);
    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        if let Err(span) = args.no_args() {
            cx.adcx().expected_no_args(span);
            return None;
        }
        Some(AttributeKind::RustcDumpDefPath(cx.attr_span))
    }
}

pub(crate) struct RustcDumpHiddenTypeOfOpaquesParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcDumpHiddenTypeOfOpaquesParser {
    const PATH: &[Symbol] = &[sym::rustc_dump_hidden_type_of_opaques];
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Crate)]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcDumpHiddenTypeOfOpaques;
}

pub(crate) struct RustcDumpInferredOutlivesParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcDumpInferredOutlivesParser {
    const PATH: &[Symbol] = &[sym::rustc_dump_inferred_outlives];
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Struct),
        Allow(Target::Enum),
        Allow(Target::Union),
        Allow(Target::TyAlias),
    ]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcDumpInferredOutlives;
}

pub(crate) struct RustcDumpItemBoundsParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcDumpItemBoundsParser {
    const PATH: &[Symbol] = &[sym::rustc_dump_item_bounds];
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::AssocTy)]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcDumpItemBounds;
}

pub(crate) struct RustcDumpLayoutParser;

impl<S: Stage> CombineAttributeParser<S> for RustcDumpLayoutParser {
    const PATH: &[Symbol] = &[sym::rustc_dump_layout];

    type Item = RustcDumpLayoutKind;

    const CONVERT: ConvertFn<Self::Item> = |items, _| AttributeKind::RustcDumpLayout(items);

    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Struct),
        Allow(Target::Enum),
        Allow(Target::Union),
        Allow(Target::TyAlias),
    ]);

    const TEMPLATE: AttributeTemplate =
        template!(List: &["abi", "align", "size", "homogenous_aggregate", "debug"]);
    fn extend(
        cx: &mut AcceptContext<'_, '_, S>,
        args: &ArgParser,
    ) -> impl IntoIterator<Item = Self::Item> {
        let Some(items) = cx.expect_list(args, cx.attr_span) else {
            return vec![];
        };

        let mut result = Vec::new();
        for item in items.mixed() {
            let Some(arg) = item.as_meta_item() else {
                cx.adcx().expected_not_literal(item.span());
                continue;
            };
            let Some(ident) = arg.ident() else {
                cx.adcx().expected_identifier(arg.span());
                return vec![];
            };
            let kind = match ident.name {
                sym::align => RustcDumpLayoutKind::Align,
                sym::backend_repr => RustcDumpLayoutKind::BackendRepr,
                sym::debug => RustcDumpLayoutKind::Debug,
                sym::homogeneous_aggregate => RustcDumpLayoutKind::HomogenousAggregate,
                sym::size => RustcDumpLayoutKind::Size,
                _ => {
                    cx.adcx().expected_specific_argument(
                        ident.span,
                        &[
                            sym::align,
                            sym::backend_repr,
                            sym::debug,
                            sym::homogeneous_aggregate,
                            sym::size,
                        ],
                    );
                    continue;
                }
            };
            result.push(kind);
        }
        result
    }
}

pub(crate) struct RustcDumpObjectLifetimeDefaultsParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcDumpObjectLifetimeDefaultsParser {
    const PATH: &[Symbol] = &[sym::rustc_dump_object_lifetime_defaults];
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::AssocConst),
        Allow(Target::AssocTy),
        Allow(Target::Const),
        Allow(Target::Enum),
        Allow(Target::Fn),
        Allow(Target::ForeignFn),
        Allow(Target::Impl { of_trait: false }),
        Allow(Target::Impl { of_trait: true }),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Method(MethodKind::Trait { body: false })),
        Allow(Target::Method(MethodKind::Trait { body: true })),
        Allow(Target::Method(MethodKind::TraitImpl)),
        Allow(Target::Struct),
        Allow(Target::Trait),
        Allow(Target::TraitAlias),
        Allow(Target::TyAlias),
        Allow(Target::Union),
    ]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcDumpObjectLifetimeDefaults;
}

pub(crate) struct RustcDumpPredicatesParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcDumpPredicatesParser {
    const PATH: &[Symbol] = &[sym::rustc_dump_predicates];
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::AssocConst),
        Allow(Target::AssocTy),
        Allow(Target::Const),
        Allow(Target::Delegation { mac: false }),
        Allow(Target::Delegation { mac: true }),
        Allow(Target::Enum),
        Allow(Target::Fn),
        Allow(Target::Impl { of_trait: false }),
        Allow(Target::Impl { of_trait: true }),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Method(MethodKind::Trait { body: false })),
        Allow(Target::Method(MethodKind::Trait { body: true })),
        Allow(Target::Method(MethodKind::TraitImpl)),
        Allow(Target::Struct),
        Allow(Target::Trait),
        Allow(Target::TraitAlias),
        Allow(Target::TyAlias),
        Allow(Target::Union),
    ]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcDumpPredicates;
}

pub(crate) struct RustcDumpSymbolNameParser;

impl<S: Stage> SingleAttributeParser<S> for RustcDumpSymbolNameParser {
    const PATH: &[Symbol] = &[sym::rustc_dump_symbol_name];
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Fn),
        Allow(Target::Method(MethodKind::TraitImpl)),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Method(MethodKind::Trait { body: true })),
        Allow(Target::ForeignFn),
        Allow(Target::ForeignStatic),
        Allow(Target::Impl { of_trait: false }),
    ]);
    const TEMPLATE: AttributeTemplate = template!(Word);
    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        if let Err(span) = args.no_args() {
            cx.adcx().expected_no_args(span);
            return None;
        }
        Some(AttributeKind::RustcDumpSymbolName(cx.attr_span))
    }
}

pub(crate) struct RustcDumpVariancesParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcDumpVariancesParser {
    const PATH: &[Symbol] = &[sym::rustc_dump_variances];
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Enum),
        Allow(Target::Fn),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Method(MethodKind::Trait { body: false })),
        Allow(Target::Method(MethodKind::Trait { body: true })),
        Allow(Target::Method(MethodKind::TraitImpl)),
        Allow(Target::Struct),
        Allow(Target::Union),
    ]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcDumpVariances;
}

pub(crate) struct RustcDumpVariancesOfOpaquesParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcDumpVariancesOfOpaquesParser {
    const PATH: &[Symbol] = &[sym::rustc_dump_variances_of_opaques];
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Crate)]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcDumpVariancesOfOpaques;
}

pub(crate) struct RustcDumpVtableParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcDumpVtableParser {
    const PATH: &[Symbol] = &[sym::rustc_dump_vtable];
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Impl { of_trait: true }),
        Allow(Target::TyAlias),
    ]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::RustcDumpVtable;
}
