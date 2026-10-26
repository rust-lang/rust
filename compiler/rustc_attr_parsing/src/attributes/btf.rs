use rustc_feature::AttributeStability;

use super::prelude::*;

fn parse_btf_tag(cx: &mut AcceptContext<'_, '_>, args: &ArgParser) -> Option<Symbol> {
    let single = cx.expect_single_element_list(args, cx.attr_span)?;
    let tag = cx.expect_string_literal(single)?;
    if tag.as_str().is_empty() {
        cx.adcx().expected_non_empty_string_literal(single.span());
        return None;
    }
    Some(tag)
}

pub(crate) struct BtfDeclTagParser;

impl CombineAttributeParser for BtfDeclTagParser {
    const PATH: &[Symbol] = &[sym::btf_decl_tag];
    type Item = Symbol;
    const CONVERT: ConvertFn<Self::Item> = |tags, _| AttributeKind::BtfDeclTag(tags);
    const ALLOWED_TARGETS: AllowedTargets<'_> = AllowedTargets::AllowList(&[
        Allow(Target::Static),
        Allow(Target::ForeignStatic),
        Allow(Target::Fn),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Method(MethodKind::Trait { body: true })),
        Allow(Target::Method(MethodKind::TraitImpl)),
        Allow(Target::Struct),
        Allow(Target::Union),
        Allow(Target::Field),
        Allow(Target::Param),
    ]);
    const TEMPLATE: AttributeTemplate = template!(List: &[r#""tag""#]);
    const STABILITY: AttributeStability = unstable!(btf_tags);

    fn extend(
        cx: &mut AcceptContext<'_, '_>,
        args: &ArgParser,
    ) -> impl IntoIterator<Item = Self::Item> {
        parse_btf_tag(cx, args)
    }
}

pub(crate) struct BtfTypeTagParser;

impl CombineAttributeParser for BtfTypeTagParser {
    const PATH: &[Symbol] = &[sym::btf_type_tag];
    type Item = Symbol;
    const CONVERT: ConvertFn<Self::Item> = |tags, _| AttributeKind::BtfTypeTag(tags);
    const ALLOWED_TARGETS: AllowedTargets<'_> = AllowedTargets::AllowList(&[
        Allow(Target::Static),
        Allow(Target::ForeignStatic),
        Allow(Target::Field),
        Allow(Target::Param),
    ]);
    const TEMPLATE: AttributeTemplate = template!(List: &[r#""tag""#]);
    const STABILITY: AttributeStability = unstable!(btf_tags);

    fn extend(
        cx: &mut AcceptContext<'_, '_>,
        args: &ArgParser,
    ) -> impl IntoIterator<Item = Self::Item> {
        parse_btf_tag(cx, args)
    }
}
