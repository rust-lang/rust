use rustc_ast::LitKind;
use rustc_feature::AttributeStability;

use super::prelude::*;

/// Parses `#[ptrauth_c_discriminator(N)]` on a static, or on a field of a type held in a static:
/// code addresses are signed with discriminator `N`.
pub(crate) struct PtrauthCDiscriminatorParser;
impl SingleAttributeParser for PtrauthCDiscriminatorParser {
    const PATH: &[Symbol] = &[sym::ptrauth_c_discriminator];
    const ALLOWED_TARGETS: AllowedTargets<'_> =
        AllowedTargets::AllowList(&[Allow(Target::Static), Allow(Target::Field)]);
    const TEMPLATE: AttributeTemplate = template!(List: &["discriminator"]);
    const STABILITY: AttributeStability = unstable!(ptrauth_c_discriminator);

    fn convert(cx: &mut AcceptContext<'_, '_>, args: &ArgParser) -> Option<AttributeKind> {
        let single = cx.expect_single_element_list(args, cx.attr_span)?;
        let Some(lit) = single.as_lit() else {
            cx.adcx().expected_integer_literal(single.span());
            return None;
        };
        let LitKind::Int(value, _) = lit.kind else {
            cx.adcx().expected_integer_literal(single.span());
            return None;
        };
        let Ok(discriminator) = u16::try_from(value.get()) else {
            cx.adcx().expected_integer_literal_in_range(single.span(), 0, u16::MAX as isize);
            return None;
        };

        Some(AttributeKind::PtrauthCDiscriminator { discriminator })
    }
}
