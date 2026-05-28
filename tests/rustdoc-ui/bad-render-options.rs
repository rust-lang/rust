// regression test for https://github.com/rust-lang/rust/issues/149187
#![deny(invalid_doc_attributes)]

#![doc(html_favicon_url)]
//~^ ERROR
//~| WARN
#![doc(html_logo_url)]
//~^ ERROR
//~| WARN
#![doc(html_playground_url)]
//~^ ERROR
//~| WARN
#![doc(issue_tracker_base_url)]
//~^ ERROR
//~| WARN
#![doc(html_favicon_url = 1)]
//~^ ERROR
//~| WARN
#![doc(html_logo_url = 2)]
//~^ ERROR
//~| WARN
#![doc(html_playground_url = 3)]
//~^ ERROR
//~| WARN
#![doc(issue_tracker_base_url = 4)]
//~^ ERROR
//~| WARN
#![doc(html_no_source = "asdf")]
//~^ ERROR
//~| WARN
