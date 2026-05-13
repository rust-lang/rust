use super::prelude::*;

pub(crate) struct MustNotSuspendParser;

impl SingleAttributeParser for MustNotSuspendParser {
    const PATH: &[rustc_span::Symbol] = &[sym::must_not_suspend];
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Struct),
        Allow(Target::Enum),
        Allow(Target::Union),
        Allow(Target::Trait),
    ]);
    const TEMPLATE: AttributeTemplate = template!(Word, List: &["count"]);

    fn convert(cx: &mut AcceptContext<'_, '_>, args: &ArgParser) -> Option<AttributeKind> {
        let reason = match args {
            ArgParser::NameValue(reason) => cx.expect_string_literal(reason),
            ArgParser::NoArgs => None,
            ArgParser::List(list) => {
                cx.adcx().expected_nv_or_no_args(list.span);
                return None;
            }
        };

        Some(AttributeKind::MustNotSupend { reason })
    }
}
