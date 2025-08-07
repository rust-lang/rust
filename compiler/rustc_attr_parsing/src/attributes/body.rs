//! Attributes that can be found in function body.

use rustc_hir::attrs::AttributeKind;
use rustc_span::{Symbol, sym};

use super::{NoArgsAttributeParser, OnDuplicate};
use crate::context::Stage;

pub(crate) struct CoroutineParser;

impl<S: Stage> NoArgsAttributeParser<S> for CoroutineParser {
    const PATH: &[Symbol] = &[sym::coroutine];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const CREATE: fn(rustc_span::Span) -> AttributeKind = |span| AttributeKind::Coroutine(span);
}
