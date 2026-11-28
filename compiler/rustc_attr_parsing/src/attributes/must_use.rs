use rustc_feature::AttributeStability;

use super::prelude::*;

pub(crate) struct MustUseParser;

impl SingleAttributeParser for MustUseParser {
    const PATH: &[Symbol] = &[sym::must_use];
    const ON_DUPLICATE: OnDuplicate = OnDuplicate::WarnButFutureError;
    const ALLOWED_TARGETS: AllowedTargets<'_> = AllowedTargets::AllowList(&[
        // tidy-alphabetical-start
        Allow(Target::Delegation { mac: false }),
        Allow(Target::Enum),
        Allow(Target::Fn),
        Allow(Target::ForeignFn),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Method(MethodKind::Trait { body: false })),
        Allow(Target::Method(MethodKind::Trait { body: true })),
        Allow(Target::Struct),
        // `impl Trait` in return position can trip
        // `unused_must_use` if `Trait` is marked as
        // `#[must_use]`
        Allow(Target::Trait),
        Allow(Target::Union),
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
        Warn(Target::Crate),
        Warn(Target::ExprField),
        Warn(Target::Expression),
        Warn(Target::ExternCrate),
        Warn(Target::Field),
        Warn(Target::ForLoop),
        Warn(Target::ForeignMod),
        Warn(Target::ForeignStatic),
        Warn(Target::GlobalAsm),
        Warn(Target::Impl { of_trait: false }),
        Warn(Target::Impl { of_trait: true }),
        Warn(Target::LifetimeParam),
        Warn(Target::Loop),
        Warn(Target::MacroCall),
        Warn(Target::MacroDef),
        Warn(Target::Method(MethodKind::TraitImpl)),
        Warn(Target::Mod),
        Warn(Target::PatField),
        Warn(Target::Statement),
        Warn(Target::Static),
        Warn(Target::TyAlias),
        Warn(Target::TypeParam),
        Warn(Target::Use),
        Warn(Target::Variant),
        Warn(Target::While),
        // tidy-alphabetical-end
    ]);
    const TEMPLATE: AttributeTemplate = template!(
        Word, NameValueStr: "reason",
        "https://doc.rust-lang.org/reference/attributes/diagnostics.html#the-must_use-attribute"
    );
    const STABILITY: AttributeStability = AttributeStability::Stable;

    fn convert(cx: &mut AcceptContext<'_, '_>, args: &ArgParser) -> Option<AttributeKind> {
        Some(AttributeKind::MustUse {
            span: cx.attr_span,
            reason: match args {
                ArgParser::NoArgs => None,
                ArgParser::NameValue(name_value) => cx.expect_string_literal(name_value),
                ArgParser::List(list) => {
                    cx.adcx().expected_nv_or_no_args(list.span);
                    return None;
                }
            },
        })
    }
}
