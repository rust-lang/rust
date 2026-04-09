use super::prelude::*;

pub(crate) struct RustcAsPtrParser;
impl<S: Stage> NoArgsAttributeParser<S> for RustcAsPtrParser {
    const PATH: &[Symbol] = &[sym::rustc_as_ptr];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const GATED: AttributeGate = gated_rustc_attr!(
        rustc_as_ptr,
        "`#[rustc_as_ptr]` is used to mark functions returning pointers to their inner allocations"
    );
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Fn),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Method(MethodKind::Trait { body: false })),
        Allow(Target::Method(MethodKind::Trait { body: true })),
        Allow(Target::Method(MethodKind::TraitImpl)),
    ]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::RustcAsPtr;
}

pub(crate) struct RustcPubTransparentParser;
impl<S: Stage> NoArgsAttributeParser<S> for RustcPubTransparentParser {
    const PATH: &[Symbol] = &[sym::rustc_pub_transparent];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const GATED: AttributeGate = gated_rustc_attr!(
        rustc_pub_transparent,
        "used internally to mark types with a `transparent` representation when it is guaranteed by the documentation"
    );
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Struct),
        Allow(Target::Enum),
        Allow(Target::Union),
    ]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::RustcPubTransparent;
}

pub(crate) struct RustcPassByValueParser;
impl<S: Stage> NoArgsAttributeParser<S> for RustcPassByValueParser {
    const PATH: &[Symbol] = &[sym::rustc_pass_by_value];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const GATED: AttributeGate = gated_rustc_attr!(
        rustc_pass_by_value,
        "`#[rustc_pass_by_value]` is used to mark types that must be passed by value instead of reference"
    );

    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Struct),
        Allow(Target::Enum),
        Allow(Target::TyAlias),
    ]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::RustcPassByValue;
}

pub(crate) struct RustcShouldNotBeCalledOnConstItemsParser;
impl<S: Stage> NoArgsAttributeParser<S> for RustcShouldNotBeCalledOnConstItemsParser {
    const PATH: &[Symbol] = &[sym::rustc_should_not_be_called_on_const_items];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const GATED: AttributeGate = gated_rustc_attr!(
        rustc_should_not_be_called_on_const_items,
        "`#[rustc_should_not_be_called_on_const_items]` is used to mark methods that don't make sense to be called on interior mutable consts"
    );

    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Method(MethodKind::TraitImpl)),
    ]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::RustcShouldNotBeCalledOnConstItems;
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
    const GATED: AttributeGate = AttributeGate::Ungated;
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::AutomaticallyDerived;
}
