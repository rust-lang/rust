//@check-pass
#![feature(rustdoc_internals)]

#[doc(rust_logo)]
//~^ WARN this attribute can only be applied at the crate level
#[doc(html_favicon_url = "example.org")]
//~^ WARN this attribute can only be applied at the crate level
#[doc(html_logo_url = "example.org")]
//~^ WARN this attribute can only be applied at the crate level
#[doc(html_playground_url = "example.org")]
//~^ WARN this attribute can only be applied at the crate level
#[doc(issue_tracker_base_url = "example.org")]
//~^ WARN this attribute can only be applied at the crate level
#[doc(html_root_url = "example.org")]
//~^ WARN this attribute can only be applied at the crate level
#[doc(html_no_source)]
//~^ WARN this attribute can only be applied at the crate level
#[doc(test(no_crate_inject))]
//~^ WARN this attribute can only be applied at the crate level
fn function() {}

fn main() {}
