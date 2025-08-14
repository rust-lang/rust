use rustc_hir::attrs::AttributeKind;
use rustc_span::{Span, Symbol, sym};

use crate::attributes::{NoArgsAttributeParser, OnDuplicate};
use crate::context::Stage;

pub(crate) struct LoopMatchParser;
impl<S: Stage> NoArgsAttributeParser<S> for LoopMatchParser {
    const PATH: &[Symbol] = &[sym::loop_match];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::LoopMatch;
}

pub(crate) struct ConstContinueParser;
impl<S: Stage> NoArgsAttributeParser<S> for ConstContinueParser {
    const PATH: &[Symbol] = &[sym::const_continue];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::ConstContinue;
}
