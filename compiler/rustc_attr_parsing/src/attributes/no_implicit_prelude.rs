use rustc_feature::AttributeStability;

use super::prelude::*;

pub(crate) struct NoImplicitPreludeParser;

impl NoArgsAttributeParser for NoImplicitPreludeParser {
    const PATH: &[rustc_span::Symbol] = &[sym::no_implicit_prelude];
    const ON_DUPLICATE: OnDuplicate = OnDuplicate::Warn;
    const ALLOWED_TARGETS: AllowedTargets<'_> = AllowedTargets::AllowList(&[
        // tidy-alphabetical-start
        Allow(Target::Crate),
        Allow(Target::Mod),
        Warn(Target::Arm),
        Warn(Target::AssocConst(AssocCtxt::Impl { of_trait: false })),
        Warn(Target::AssocConst(AssocCtxt::Impl { of_trait: true })),
        Warn(Target::AssocConst(AssocCtxt::Trait)),
        Warn(Target::AssocTy(AssocCtxt::Impl { of_trait: false })),
        Warn(Target::AssocTy(AssocCtxt::Impl { of_trait: true })),
        Warn(Target::AssocTy(AssocCtxt::Trait)),
        Warn(Target::Break),
        Warn(Target::Closure),
        Warn(Target::Const),
        Warn(Target::ConstParam),
        Warn(Target::Enum),
        Warn(Target::ExprField),
        Warn(Target::Expression),
        Warn(Target::ExternCrate),
        Warn(Target::Field),
        Warn(Target::Fn),
        Warn(Target::ForLoop),
        Warn(Target::ForeignFn),
        Warn(Target::ForeignMod),
        Warn(Target::ForeignStatic),
        Warn(Target::GlobalAsm),
        Warn(Target::Impl { of_trait: false }),
        Warn(Target::Impl { of_trait: true }),
        Warn(Target::LifetimeParam),
        Warn(Target::Loop),
        Warn(Target::MacroCall),
        Warn(Target::MacroDef),
        Warn(Target::Method(MethodKind::Inherent)),
        Warn(Target::Method(MethodKind::Trait { body: false })),
        Warn(Target::Method(MethodKind::Trait { body: true })),
        Warn(Target::Method(MethodKind::TraitImpl)),
        Warn(Target::PatField),
        Warn(Target::Statement),
        Warn(Target::Static),
        Warn(Target::Struct),
        Warn(Target::Trait),
        Warn(Target::TyAlias),
        Warn(Target::TypeParam),
        Warn(Target::Union),
        Warn(Target::Use),
        Warn(Target::Variant),
        Warn(Target::While),
        // tidy-alphabetical-end
    ]);
    const STABILITY: AttributeStability = AttributeStability::Stable;
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::NoImplicitPrelude;
}
