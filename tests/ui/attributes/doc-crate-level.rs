#[doc(html_favicon_url = "example.org")]
//~^ ERROR this attribute can only be applied at the crate level
#[doc(html_logo_url = "example.org")]
//~^ ERROR this attribute can only be applied at the crate level
#[doc(html_playground_url = "example.org")]
//~^ ERROR this attribute can only be applied at the crate level
#[doc(issue_tracker_base_url = "example.org")]
//~^ ERROR this attribute can only be applied at the crate level
#[doc(html_root_url = "example.org")]
//~^ ERROR this attribute can only be applied at the crate level
#[doc(html_no_source)]
//~^ ERROR this attribute can only be applied at the crate level
#[doc(test(no_crate_inject))]
//~^ ERROR this attribute can only be applied at the crate level
fn function() {}

fn main() {}
