use rustc_attr_ir::{ReprAttr, find_attr};
use rustc_feature::AttributeStability;

use super::prelude::*;
use crate::diagnostics::{InvalidCBufferLength, RustcPubTransparent};

pub(crate) struct LintAsPtrParser;
impl NoArgsAttributeParser for LintAsPtrParser {
    const PATH: &[Symbol] = &[sym::lint, sym::as_ptr];
    const ALLOWED_TARGETS: AllowedTargets<'_> = AllowedTargets::AllowList(&[
        Allow(Target::Fn),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Method(MethodKind::Trait { body: false })),
        Allow(Target::Method(MethodKind::Trait { body: true })),
        Allow(Target::Method(MethodKind::TraitImpl)),
    ]);
    const STABILITY: AttributeStability = unstable!(rustc_attrs);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::LintAsPtr;
}

pub(crate) struct LintCBufferLengthParser;

impl SingleAttributeParser for LintCBufferLengthParser {
    const PATH: &[Symbol] = &[sym::lint, sym::c_buffer_length];
    const ALLOWED_TARGETS: AllowedTargets<'_> =
        AllowedTargets::AllowList(&[Allow(Target::Fn), Allow(Target::ForeignFn)]);
    const TEMPLATE: AttributeTemplate = template!(List: &["pointer, length"]);
    const STABILITY: AttributeStability = unstable!(lint_c_buffer_length);

    fn convert(cx: &mut AcceptContext<'_, '_>, args: &ArgParser) -> Option<AttributeKind> {
        let parsed = try {
            let [pointer, length] = args.as_list()?.sub_parsers().as_array()?;
            let pointer = pointer.meta_item_no_args()?.ident()?;
            let length = length.meta_item_no_args()?.ident()?;
            (pointer.normalize_to_macro_rules() != length.normalize_to_macro_rules())
                .then_some((pointer, length))?
        };
        let Some((pointer, length)) = parsed else {
            args.ignore_args();
            cx.emit_err(InvalidCBufferLength { span: cx.attr_span });
            return None;
        };

        Some(AttributeKind::LintCBufferLength { pointer, length })
    }
}

pub(crate) struct LintNeverReturnsNullPtrParser;

impl NoArgsAttributeParser for LintNeverReturnsNullPtrParser {
    const PATH: &[Symbol] = &[sym::lint, sym::never_returns_null_ptr];
    const ALLOWED_TARGETS: AllowedTargets<'_> = AllowedTargets::AllowList(&[
        Allow(Target::Fn),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Method(MethodKind::Trait { body: false })),
        Allow(Target::Method(MethodKind::Trait { body: true })),
        Allow(Target::Method(MethodKind::TraitImpl)),
    ]);
    const STABILITY: AttributeStability = unstable!(rustc_attrs);

    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::LintNeverReturnsNullPtr;
}

pub(crate) struct RustcPubTransparentParser;
impl NoArgsAttributeParser for RustcPubTransparentParser {
    const PATH: &[Symbol] = &[sym::rustc_pub_transparent];
    const ALLOWED_TARGETS: AllowedTargets<'_> = AllowedTargets::AllowList(&[
        Allow(Target::Struct),
        Allow(Target::Enum),
        Allow(Target::Union),
    ]);
    const STABILITY: AttributeStability = unstable!(rustc_attrs);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::RustcPubTransparent;

    fn finalize_check(cx: &mut FinalizeCheckContext<'_, '_>, attr_span: Span) {
        // `#[rustc_pub_transparent]` may only be applied to `#[repr(transparent)]` types.
        let is_transparent = find_attr!(
            cx.parsed_attrs,
            Repr { reprs, .. } if reprs.iter().any(|(r, _)| r == &ReprAttr::ReprTransparent)
        );
        if !is_transparent {
            cx.emit_err(RustcPubTransparent { span: cx.target_span, attr_span });
        }
    }
}

pub(crate) struct LintPassByValueParser;
impl NoArgsAttributeParser for LintPassByValueParser {
    const PATH: &[Symbol] = &[sym::lint, sym::pass_by_value];
    const ALLOWED_TARGETS: AllowedTargets<'_> = AllowedTargets::AllowList(&[
        Allow(Target::Struct),
        Allow(Target::Enum),
        Allow(Target::TyAlias),
    ]);
    const STABILITY: AttributeStability = unstable!(rustc_attrs);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::LintPassByValue;
}

pub(crate) struct LintShouldNotBeCalledOnConstItemsParser;
impl NoArgsAttributeParser for LintShouldNotBeCalledOnConstItemsParser {
    const PATH: &[Symbol] = &[sym::lint, sym::should_not_be_called_on_const_items];
    const ALLOWED_TARGETS: AllowedTargets<'_> = AllowedTargets::AllowList(&[
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Method(MethodKind::TraitImpl)),
    ]);
    const STABILITY: AttributeStability = unstable!(rustc_attrs);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::LintShouldNotBeCalledOnConstItems;
}

pub(crate) struct AutomaticallyDerivedParser;
impl NoArgsAttributeParser for AutomaticallyDerivedParser {
    const PATH: &[Symbol] = &[sym::automatically_derived];
    const ON_DUPLICATE: OnDuplicate = OnDuplicate::Warn;
    const ALLOWED_TARGETS: AllowedTargets<'_> = AllowedTargets::AllowList(&[
        // tidy-alphabetical-start
        Allow(Target::Impl { of_trait: true }),
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
        Warn(Target::LifetimeParam),
        Warn(Target::Loop),
        Warn(Target::MacroCall),
        Warn(Target::MacroDef),
        Warn(Target::Method(MethodKind::Inherent)),
        Warn(Target::Method(MethodKind::Trait { body: false })),
        Warn(Target::Method(MethodKind::Trait { body: true })),
        Warn(Target::Method(MethodKind::TraitImpl)),
        Warn(Target::Mod),
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
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::AutomaticallyDerived;
}
