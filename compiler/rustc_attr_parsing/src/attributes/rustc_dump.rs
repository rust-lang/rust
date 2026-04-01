use rustc_hir::attrs::AttributeKind;
use rustc_hir::{MethodKind, Target};
use rustc_span::{Span, Symbol, sym};

use crate::attributes::prelude::Allow;
use crate::attributes::{NoArgsAttributeParser, OnDuplicate};
use crate::context::Stage;
use crate::target_checking::AllowedTargets;

pub(crate) struct RustcDumpUserArgsParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcDumpUserArgsParser {
    const PATH: &[Symbol] = &[sym::rustc_dump_user_args];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Fn)]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcDumpUserArgs;
}

pub(crate) struct RustcDumpDefParentsParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcDumpDefParentsParser {
    const PATH: &[Symbol] = &[sym::rustc_dump_def_parents];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Fn)]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcDumpDefParents;
}

pub(crate) struct RustcDumpInferredOutlivesParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcDumpInferredOutlivesParser {
    const PATH: &[Symbol] = &[sym::rustc_dump_inferred_outlives];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
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
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::AssocTy)]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcDumpItemBounds;
}

pub(crate) struct RustcDumpObjectLifetimeDefaultsParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcDumpObjectLifetimeDefaultsParser {
    const PATH: &[Symbol] = &[sym::rustc_dump_object_lifetime_defaults];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
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
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
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

pub(crate) struct RustcDumpVariancesParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcDumpVariancesParser {
    const PATH: &[Symbol] = &[sym::rustc_dump_variances];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
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
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Crate)]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcDumpVariancesOfOpaques;
}

pub(crate) struct RustcDumpVtableParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcDumpVtableParser {
    const PATH: &[Symbol] = &[sym::rustc_dump_vtable];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Impl { of_trait: true }),
        Allow(Target::TyAlias),
    ]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::RustcDumpVtable;
}
