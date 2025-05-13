/// This struct is needed to enforce `#[must_use]` on [tracing::span::EnteredSpan]
/// while wrapping them in an `Option`.
#[must_use]
pub enum MaybeEnteredSpan {
    Some(tracing::span::EnteredSpan),
    None,
}

#[macro_export]
macro_rules! enter_trace_span {
    ($machine:ident, $($tt:tt)*) => {
        if $machine::TRACING_ENABLED {
            $crate::interpret::tracing_utils::MaybeEnteredSpan::Some(tracing::info_span!($($tt)*).entered())
        } else {
            $crate::interpret::tracing_utils::MaybeEnteredSpan::None
        }
    }
}
