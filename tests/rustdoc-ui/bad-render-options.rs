// regression test for https://github.com/rust-lang/rust/issues/149187

#![doc(html_favicon_url)]
//~^ ERROR
#![doc(html_logo_url)]
//~^ ERROR
#![doc(html_playground_url)]
//~^ ERROR
#![doc(issue_tracker_base_url)]
//~^ ERROR
#![doc(html_favicon_url = 1)]
//~^ ERROR
#![doc(html_logo_url = 2)]
//~^ ERROR
#![doc(html_playground_url = 3)]
//~^ ERROR
#![doc(issue_tracker_base_url = 4)]
//~^ ERROR
#![doc(html_no_source = "asdf")]
//~^ ERROR
