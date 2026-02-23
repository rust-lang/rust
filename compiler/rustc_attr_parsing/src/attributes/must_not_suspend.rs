use super::prelude::*;

pub(crate) struct MustNotSuspendParser;

impl<S: Stage> SingleAttributeParser<S> for MustNotSuspendParser {
    const PATH: &[rustc_span::Symbol] = &[sym::must_not_suspend];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepInnermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Struct),
        Allow(Target::Enum),
        Allow(Target::Union),
        Allow(Target::Trait),
    ]);
    const TEMPLATE: AttributeTemplate = template!(Word, List: &["count"]);

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        let reason = match args {
            ArgParser::NameValue(reason) => match reason.value_as_str() {
                Some(val) => Some(val),
                None => {
                    cx.expected_nv_or_no_args(reason.value_span);
                    return None;
                }
            },
            ArgParser::NoArgs => None,
            ArgParser::List(list) => {
                cx.expected_nv_or_no_args(list.span);
                return None;
            }
        };

        Some(AttributeKind::MustNotSupend { reason })
    }
}
