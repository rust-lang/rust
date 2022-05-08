use std::borrow::Borrow;

use rustc_data_structures::profiling::EventArgRecorder;

/// Extension trait for self-profiling purposes: allows to record spans within a generic activity's
/// event arguments.
pub trait SpannedEventArgRecorder {
    /// Records the following event arguments within the current generic activity being profiled:
    /// - the provided `event_arg`
    /// - a string representation of the provided `span`
    ///
    /// Note: when self-profiling with costly event arguments, at least one argument
    /// needs to be recorded. A panic will be triggered if that doesn't happen.
    fn record_arg_with_span<A>(&mut self, event_arg: A, span: crate::Span)
    where
        A: Borrow<str> + Into<String>;
}

impl SpannedEventArgRecorder for EventArgRecorder<'_> {
    fn record_arg_with_span<A>(&mut self, event_arg: A, span: crate::Span)
    where
        A: Borrow<str> + Into<String>,
    {
        self.record_arg(event_arg);

        let span_arg = crate::with_session_globals(|session_globals| {
            if let Some(source_map) = &*session_globals.source_map.borrow() {
                source_map.span_to_embeddable_string(span)
            } else {
                format!("{:?}", span)
            }
        });
        self.record_arg(span_arg);
    }
}
