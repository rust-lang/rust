extern crate span_join;

// This macro invocation will panic if it successfully joins 2 spans.
span_join::different_hygiene!(token);
