#![feature(rustdoc_internals)]

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

fn main() {}
