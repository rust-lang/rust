use rustc_ast::LitKind;
use rustc_attr_data_structures::AttributeKind;
use rustc_feature::{AttributeTemplate, template};
use rustc_middle::bug;
use rustc_span::sym;

use super::{AcceptContext, AttributeOrder, OnDuplicate};
use crate::attributes::SingleAttributeParser;
use crate::context::Stage;
use crate::parser::ArgParser;

pub(crate) struct EiiParser;

impl<S: Stage> SingleAttributeParser<S> for EiiParser {
    const PATH: &'static [rustc_span::Symbol] = &[sym::eii];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepLast;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::WarnButFutureError;
    const TEMPLATE: AttributeTemplate = template!(Word);

    fn convert(cx: &AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        Some(AttributeKind::Eii(cx.attr_span))
    }
}

// pub(crate) struct EiiImplParser;
//
// impl<S: Stage> SingleAttributeParser<S> for EiiImplParser {
//     const PATH: &'static [rustc_span::Symbol] = &[sym::eii_impl];
//     const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepLast;
//     const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::WarnButFutureError;
//     const TEMPLATE: AttributeTemplate = template!(List: "<eii_id>, /*opt*/ default");
//
//     fn convert(cx: &AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
//         let Some(l) = args.list() else {
//             cx.expected_list(cx.attr_span);
//             return None;
//         };
//
//         let mut args = l.mixed();
//
//         let id = args.next().unwrap();
//         let is_default = args.next();
//         assert!(args.next().is_none());
//
//         let Some(id) = id.lit().and_then(|i| if let LitKind::Int(i, _) = i.kind {
//             Some(i)
//         } else {
//             None
//         }) else {
//             bug!("expected integer");
//         };
//
//         let Ok(id) = u32::try_from(id.get()) else {
//             bug!("too large");
//         };
//         let id = EiiId::from(id);
//
//         // AttributeKind::EiiImpl { eii: (), is_default: () }
//         todo!()
//     }
// }
//
