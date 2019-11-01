// build-pass (FIXME(62277): could be check-pass?)

#[doc]
//~^ WARN attribute must be of the form
//~| WARN this was previously accepted
#[ignore()]
//~^ WARN attribute must be of the form
//~| WARN this was previously accepted
#[inline = ""]
//~^ WARN attribute must be of the form
//~| WARN this was previously accepted
#[link]
//~^WARN attribute must be of the form
//~| WARN this was previously accepted
#[link = ""]
//~^ WARN attribute must be of the form
//~| WARN this was previously accepted
fn main() {}
