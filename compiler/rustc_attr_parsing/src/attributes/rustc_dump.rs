use rustc_hir::Target;
use rustc_hir::attrs::AttributeKind;
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

pub(crate) struct RustcDumpItemBoundsParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcDumpItemBoundsParser {
    const PATH: &[Symbol] = &[sym::rustc_dump_item_bounds];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::AssocTy)]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcDumpItemBounds;
}

pub(crate) struct RustcDumpPredicatesParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcDumpPredicatesParser {
    const PATH: &[Symbol] = &[sym::rustc_dump_predicates];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Struct),
        Allow(Target::Enum),
        Allow(Target::Union),
        Allow(Target::Trait),
        Allow(Target::AssocTy),
    ]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcDumpPredicates;
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
