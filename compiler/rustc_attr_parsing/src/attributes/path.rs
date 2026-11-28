use rustc_feature::AttributeStability;

use super::prelude::*;

pub(crate) struct PathParser;

impl SingleAttributeParser for PathParser {
    const PATH: &[Symbol] = &[sym::path];
    const ON_DUPLICATE: OnDuplicate = OnDuplicate::WarnButFutureError;
    const ALLOWED_TARGETS: AllowedTargets<'_> = AllowedTargets::AllowList(&[
        // tidy-alphabetical-start
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
    const TEMPLATE: AttributeTemplate = template!(
        NameValueStr: "file",
        "https://doc.rust-lang.org/reference/items/modules.html#the-path-attribute"
    );
    const STABILITY: AttributeStability = AttributeStability::Stable;

    fn convert(cx: &mut AcceptContext<'_, '_>, args: &ArgParser) -> Option<AttributeKind> {
        let nv = cx.expect_name_value(args, cx.attr_span, None)?;
        let path = cx.expect_string_literal(nv)?;

        Some(AttributeKind::Path(path, cx.attr_span))
    }
}
