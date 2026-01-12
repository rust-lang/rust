use rustc_hir::Target;
use rustc_hir::attrs::AttributeKind;
use rustc_span::{Span, Symbol, sym};

use crate::attributes::prelude::Allow;
use crate::attributes::{NoArgsAttributeParser, OnDuplicate};
use crate::context::Stage;
use crate::target_checking::AllowedTargets;

pub(crate) struct RustcDumpUserArgs;

impl<S: Stage> NoArgsAttributeParser<S> for RustcDumpUserArgs {
    const PATH: &[Symbol] = &[sym::rustc_dump_user_args];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Fn)]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcDumpUserArgs;
}

pub(crate) struct RustcDumpDefParents;

impl<S: Stage> NoArgsAttributeParser<S> for RustcDumpDefParents {
    const PATH: &[Symbol] = &[sym::rustc_dump_def_parents];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Fn)]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcDumpDefParents;
}

pub(crate) struct RustcDumpItemBounds;

impl<S: Stage> NoArgsAttributeParser<S> for RustcDumpItemBounds {
    const PATH: &[Symbol] = &[sym::rustc_dump_item_bounds];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::AssocTy)]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcDumpItemBounds;
}

pub(crate) struct RustcDumpVtable;

impl<S: Stage> NoArgsAttributeParser<S> for RustcDumpVtable {
    const PATH: &[Symbol] = &[sym::rustc_dump_vtable];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Impl { of_trait: true }),
        Allow(Target::TyAlias),
    ]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::RustcDumpVtable;
}
