use super::prelude::*;

pub(crate) struct MustUseParser;

impl SingleAttributeParser for MustUseParser {
    const PATH: &[Symbol] = &[sym::must_use];
    const ON_DUPLICATE: OnDuplicate = OnDuplicate::WarnButFutureError;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowListWarnRest(&[
        Allow(Target::Fn),
        Allow(Target::Enum),
        Allow(Target::Struct),
        Allow(Target::Union),
        Allow(Target::Method(MethodKind::Trait { body: false })),
        Allow(Target::Method(MethodKind::Trait { body: true })),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::ForeignFn),
        // `impl Trait` in return position can trip
        // `unused_must_use` if `Trait` is marked as
        // `#[must_use]`
        Allow(Target::Trait),
        Error(Target::WherePredicate),
    ]);
    const TEMPLATE: AttributeTemplate = template!(
        Word, NameValueStr: "reason",
        "https://doc.rust-lang.org/reference/attributes/diagnostics.html#the-must_use-attribute"
    );

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
