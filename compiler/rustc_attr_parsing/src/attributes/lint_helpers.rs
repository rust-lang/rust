use super::prelude::*;

pub(crate) struct RustcAsPtrParser;
impl NoArgsAttributeParser for RustcAsPtrParser {
    const PATH: &[Symbol] = &[sym::rustc_as_ptr];
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
impl NoArgsAttributeParser for RustcPubTransparentParser {
    const PATH: &[Symbol] = &[sym::rustc_pub_transparent];
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Struct),
        Allow(Target::Enum),
        Allow(Target::Union),
    ]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::RustcPubTransparent;
}

pub(crate) struct RustcPassByValueParser;
impl NoArgsAttributeParser for RustcPassByValueParser {
    const PATH: &[Symbol] = &[sym::rustc_pass_by_value];
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Struct),
        Allow(Target::Enum),
        Allow(Target::TyAlias),
    ]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::RustcPassByValue;
}

pub(crate) struct RustcShouldNotBeCalledOnConstItemsParser;
impl NoArgsAttributeParser for RustcShouldNotBeCalledOnConstItemsParser {
    const PATH: &[Symbol] = &[sym::rustc_should_not_be_called_on_const_items];
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Method(MethodKind::TraitImpl)),
    ]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::RustcShouldNotBeCalledOnConstItems;
}

pub(crate) struct AutomaticallyDerivedParser;
impl NoArgsAttributeParser for AutomaticallyDerivedParser {
    const PATH: &[Symbol] = &[sym::automatically_derived];
    const ON_DUPLICATE: OnDuplicate = OnDuplicate::Warn;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowListWarnRest(&[
        Allow(Target::Impl { of_trait: true }),
        Error(Target::Crate),
        Error(Target::WherePredicate),
    ]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::AutomaticallyDerived;
}
