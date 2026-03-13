use std::str::FromStr;

use rustc_ast::LitKind;
use rustc_ast::expand::autodiff_attrs::{DiffActivity, DiffMode};
use rustc_feature::{AttributeTemplate, template};
use rustc_hir::attrs::{AttributeKind, RustcAutodiff};
use rustc_hir::{MethodKind, Target};
use rustc_span::{Symbol, sym};
use thin_vec::ThinVec;

use crate::attributes::prelude::Allow;
use crate::attributes::{AttributeOrder, OnDuplicate, SingleAttributeParser};
use crate::context::{AcceptContext, Stage};
use crate::parser::{ArgParser, MetaItemOrLitParser};
use crate::target_checking::AllowedTargets;

pub(crate) struct RustcAutodiffParser;

impl<S: Stage> SingleAttributeParser<S> for RustcAutodiffParser {
    const PATH: &[Symbol] = &[sym::rustc_autodiff];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepInnermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Fn),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Method(MethodKind::Trait { body: true })),
        Allow(Target::Method(MethodKind::TraitImpl)),
    ]);
    const TEMPLATE: AttributeTemplate = template!(
        List: &["MODE", "WIDTH", "INPUT_ACTIVITIES", "OUTPUT_ACTIVITY"],
        "https://doc.rust-lang.org/std/autodiff/index.html"
    );

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        let list = match args {
            ArgParser::NoArgs => return Some(AttributeKind::RustcAutodiff(None)),
            ArgParser::List(list) => list,
            ArgParser::NameValue(_) => {
                cx.expected_list_or_no_args(cx.attr_span);
                return None;
            }
        };

        let mut items = list.mixed().peekable();

        // Parse name
        let Some(mode) = items.next() else {
            cx.expected_at_least_one_argument(list.span);
            return None;
        };
        let Some(mode) = mode.meta_item() else {
            cx.expected_identifier(mode.span());
            return None;
        };
        let Ok(()) = mode.args().no_args() else {
            cx.expected_identifier(mode.span());
            return None;
        };
        let Some(mode) = mode.path().word() else {
            cx.expected_identifier(mode.span());
            return None;
        };
        let Ok(mode) = DiffMode::from_str(mode.as_str()) else {
            cx.expected_specific_argument(mode.span, DiffMode::all_modes());
            return None;
        };

        // Parse width
        let width = if let Some(width) = items.peek()
            && let MetaItemOrLitParser::Lit(width) = width
            && let LitKind::Int(width, _) = width.kind
            && let Ok(width) = width.0.try_into()
        {
            _ = items.next();
            width
        } else {
            1
        };

        // Parse activities
        let mut activities = ThinVec::new();
        for activity in items {
            let MetaItemOrLitParser::MetaItemParser(activity) = activity else {
                cx.expected_specific_argument(activity.span(), DiffActivity::all_activities());
                return None;
            };
            let Ok(()) = activity.args().no_args() else {
                cx.expected_specific_argument(activity.span(), DiffActivity::all_activities());
                return None;
            };
            let Some(activity) = activity.path().word() else {
                cx.expected_specific_argument(activity.span(), DiffActivity::all_activities());
                return None;
            };
            let Ok(activity) = DiffActivity::from_str(activity.as_str()) else {
                cx.expected_specific_argument(activity.span, DiffActivity::all_activities());
                return None;
            };

            activities.push(activity);
        }
        let Some(ret_activity) = activities.pop() else {
            cx.expected_specific_argument(
                list.span.with_lo(list.span.hi()),
                DiffActivity::all_activities(),
            );
            return None;
        };

        Some(AttributeKind::RustcAutodiff(Some(Box::new(RustcAutodiff {
            mode,
            width,
            input_activity: activities,
            ret_activity,
        }))))
    }
}
