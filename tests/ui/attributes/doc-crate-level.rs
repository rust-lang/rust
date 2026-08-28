#![feature(rustdoc_internals)]
#![doc(fake_variadic)]
//~^ ERROR `#![doc(fake_variadic = "...")]` isn't allowed as a crate-level attribute
#![doc(alias = "test")]
//~^ ERROR `#![doc(alias = "...")]` isn't allowed as a crate-level attribute
#![doc(search_unbox)]
//~^ ERROR `#![doc(search_unbox = "...")]` isn't allowed as a crate-level attribute

#[doc(rust_logo)]
//~^ ERROR crate-level attribute should be an inner attribute: add an exclamation mark: `#![doc]`
#[doc(html_favicon_url = "example.org")]
//~^ ERROR crate-level attribute should be an inner attribute: add an exclamation mark: `#![doc]`
#[doc(html_logo_url = "example.org")]
//~^ ERROR crate-level attribute should be an inner attribute: add an exclamation mark: `#![doc]`
#[doc(html_playground_url = "example.org")]
//~^ ERROR crate-level attribute should be an inner attribute: add an exclamation mark: `#![doc]`
#[doc(issue_tracker_base_url = "example.org")]
//~^ ERROR crate-level attribute should be an inner attribute: add an exclamation mark: `#![doc]`
#[doc(html_root_url = "example.org")]
//~^ ERROR crate-level attribute should be an inner attribute: add an exclamation mark: `#![doc]`
#[doc(html_no_source)]
//~^ ERROR crate-level attribute should be an inner attribute: add an exclamation mark: `#![doc]`
#[doc(test(no_crate_inject))]
//~^ ERROR crate-level attribute should be an inner attribute: add an exclamation mark: `#![doc]`
fn function() {}

#![doc(rust_logo)]
//~^ ERROR an inner attribute is not permitted in this context
#![doc(html_favicon_url = "example.org")]
//~^ ERROR an inner attribute is not permitted in this context
#![doc(html_logo_url = "example.org")]
//~^ ERROR an inner attribute is not permitted in this context
#![doc(html_playground_url = "example.org")]
//~^ ERROR an inner attribute is not permitted in this context
#![doc(issue_tracker_base_url = "example.org")]
//~^ ERROR an inner attribute is not permitted in this context
#![doc(html_root_url = "example.org")]
//~^ ERROR an inner attribute is not permitted in this context
#![doc(html_no_source)]
//~^ ERROR an inner attribute is not permitted in this context
#![doc(test(no_crate_inject))]
//~^ ERROR an inner attribute is not permitted in this context
fn main() {}
