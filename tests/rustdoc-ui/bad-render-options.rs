// regression test for https://github.com/rust-lang/rust/issues/149187

#![doc(html_favicon_url)]
//~^ ERROR: malformed `doc` attribute
//~| NOTE expected this to be of the form `html_favicon_url = "..."`
#![doc(html_logo_url)]
//~^ ERROR: malformed `doc` attribute
//~| NOTE expected this to be of the form `html_logo_url = "..."`
#![doc(html_playground_url)]
//~^ ERROR: malformed `doc` attribute
//~| NOTE expected this to be of the form `html_playground_url = "..."`
#![doc(issue_tracker_base_url)]
//~^ ERROR: malformed `doc` attribute
//~| NOTE expected this to be of the form `issue_tracker_base_url = "..."`
#![doc(html_favicon_url = 1)]
//~^ ERROR malformed `doc` attribute
//~| NOTE expected a string literal
#![doc(html_logo_url = 2)]
//~^ ERROR malformed `doc` attribute
//~| NOTE expected a string literal
#![doc(html_playground_url = 3)]
//~^ ERROR malformed `doc` attribute
//~| NOTE expected a string literal
#![doc(issue_tracker_base_url = 4)]
//~^ ERROR malformed `doc` attribute
//~| NOTE expected a string literal
#![doc(html_no_source = "asdf")]
//~^ ERROR malformed `doc` attribute
//~| NOTE didn't expect any arguments here
