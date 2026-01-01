use rustc_feature::{AttributeTemplate, template};
use rustc_hir::attrs::{AttributeKind, OnMoveAttrArg, OnMoveAttribute};
use rustc_hir::lints::AttributeLintKind;
use rustc_parse_format::{ParseMode, Parser, Piece, Position};
use rustc_session::lint::builtin::{
    MALFORMED_DIAGNOSTIC_ATTRIBUTES, MALFORMED_DIAGNOSTIC_FORMAT_LITERALS,
};
use rustc_span::{InnerSpan, Span, Symbol, kw, sym};
use thin_vec::ThinVec;

use crate::attributes::{AttributeOrder, OnDuplicate, SingleAttributeParser};
use crate::context::{AcceptContext, Stage};
use crate::parser::ArgParser;
use crate::target_checking::{ALL_TARGETS, AllowedTargets};

pub(crate) struct OnMoveParser;

impl<S: Stage> SingleAttributeParser<S> for OnMoveParser {
    const PATH: &[Symbol] = &[sym::diagnostic, sym::on_move];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepInnermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(ALL_TARGETS); // Check in check_attr.
    const TEMPLATE: AttributeTemplate = template!(List: &["message", "label"]);

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        let get_format_ranges = |cx: &mut AcceptContext<'_, '_, S>,
                                 symbol: &Symbol,
                                 span: Span|
         -> ThinVec<(usize, usize)> {
            let mut parser = Parser::new(symbol.as_str(), None, None, false, ParseMode::Diagnostic);
            let pieces: Vec<_> = parser.by_ref().collect();
            let mut spans = ThinVec::new();

            for piece in pieces {
                match piece {
                    Piece::NextArgument(arg) => match arg.position {
                        Position::ArgumentNamed(name) if name == kw::SelfUpper.as_str() => {
                            spans.push((arg.position_span.start - 2, arg.position_span.end));
                        }
                        Position::ArgumentNamed(name) => {
                            let span = span.from_inner(InnerSpan {
                                start: arg.position_span.start,
                                end: arg.position_span.end,
                            });
                            cx.emit_lint(
                                MALFORMED_DIAGNOSTIC_FORMAT_LITERALS,
                                AttributeLintKind::OnMoveMalformedFormatLiterals {
                                    name: Symbol::intern(name),
                                },
                                span,
                            )
                        }
                        _ => {}
                    },
                    _ => continue,
                }
            }
            spans
        };
        let Some(list) = args.list() else {
            cx.expected_list(cx.attr_span, args);
            return None;
        };

        if list.is_empty() {
            cx.emit_lint(
                MALFORMED_DIAGNOSTIC_ATTRIBUTES,
                AttributeLintKind::OnMoveMalformedAttrExpectedLiteralOrDelimiter,
                list.span,
            );
            return None;
        }

        let mut message = None;
        let mut label = None;

        for item in list.mixed() {
            let Some(item) = item.meta_item() else {
                cx.expected_specific_argument(item.span(), &[sym::message, sym::label]);
                return None;
            };

            let Some(name_value) = item.args().name_value() else {
                cx.expected_name_value(cx.attr_span, item.path().word_sym());
                return None;
            };

            let Some(value) = name_value.value_as_str() else {
                cx.expected_string_literal(name_value.value_span, None);
                return None;
            };

            let value_span = name_value.value_span;
            if item.path().word_is(sym::message) && message.is_none() {
                let spans = get_format_ranges(cx, &value, value_span);
                message = Some(OnMoveAttrArg::new(value, spans));
                continue;
            } else if item.path().word_is(sym::label) && label.is_none() {
                let spans = get_format_ranges(cx, &value, value_span);
                label = Some(OnMoveAttrArg::new(value, spans));
                continue;
            }

            cx.emit_lint(
                MALFORMED_DIAGNOSTIC_ATTRIBUTES,
                AttributeLintKind::OnMoveMalformedAttr,
                item.span(),
            )
        }
        Some(AttributeKind::OnMove(Box::new(OnMoveAttribute {
            span: cx.attr_span,
            message,
            label,
        })))
    }
}
