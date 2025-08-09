use rustc_hir::attrs::AttributeKind;
use rustc_hir::{MethodKind, Target};
use rustc_span::{Span, Symbol, sym};

use crate::attributes::{NoArgsAttributeParser, OnDuplicate};
use crate::context::MaybeWarn::{Allow, Error};
use crate::context::{AllowedTargets, Stage};
pub(crate) struct AsPtrParser;
impl<S: Stage> NoArgsAttributeParser<S> for AsPtrParser {
    const PATH: &[Symbol] = &[sym::rustc_as_ptr];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Fn),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Method(MethodKind::Trait { body: false })),
        Allow(Target::Method(MethodKind::Trait { body: true })),
        Allow(Target::Method(MethodKind::TraitImpl)),
    ]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::AsPtr;
}

pub(crate) struct PubTransparentParser;
impl<S: Stage> NoArgsAttributeParser<S> for PubTransparentParser {
    const PATH: &[Symbol] = &[sym::rustc_pub_transparent];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Struct),
        Allow(Target::Enum),
        Allow(Target::Union),
    ]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::PubTransparent;
}

pub(crate) struct PassByValueParser;
impl<S: Stage> NoArgsAttributeParser<S> for PassByValueParser {
    const PATH: &[Symbol] = &[sym::rustc_pass_by_value];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Struct),
        Allow(Target::Enum),
        Allow(Target::TyAlias),
    ]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::PassByValue;
}

pub(crate) struct AutomaticallyDerivedParser;
impl<S: Stage> NoArgsAttributeParser<S> for AutomaticallyDerivedParser {
    const PATH: &[Symbol] = &[sym::automatically_derived];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowListWarnRest(&[
        Allow(Target::Impl { of_trait: true }),
        Error(Target::Crate),
        Error(Target::WherePredicate),
    ]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::AutomaticallyDerived;
}
