//@ check-fail

#![deny(rustdoc::bare_urls)]

// examples of bare urls that are beyond our ability to generate suggestions for

// this falls through every heuristic in `source_span_for_markdown_range`,
// and thus does not get any suggestion.
#[doc = "good: <https://example.com/> \n\n"]
//~^ ERROR this URL is not a hyperlink
#[doc = "bad: https://example.com/"]
pub fn duplicate_raw() {}
