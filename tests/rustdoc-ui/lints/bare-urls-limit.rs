//@ check-fail

#![deny(rustdoc::bare_urls)]

#[doc = "good: <https://example.com/> \n\n"]
#[doc = "bad: https://example.com/"]
//~^ ERROR this URL is not a hyperlink
pub fn duplicate_raw() {}
