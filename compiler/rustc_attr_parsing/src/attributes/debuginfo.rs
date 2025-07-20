use rustc_hir::attrs::AttributeKind;
use rustc_span::{Span, Symbol, sym};

use crate::attributes::{NoArgsAttributeParser, OnDuplicate};
use crate::context::Stage;

pub(crate) struct DebuginfoTransparentParser;
impl<S: Stage> NoArgsAttributeParser<S> for DebuginfoTransparentParser {
    const PATH: &[Symbol] = &[sym::debuginfo_transparent];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::DebuginfoTransparent;
}
