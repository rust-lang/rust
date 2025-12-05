//@ revisions: in_attr in_flag
//@[in_flag] compile-flags: -l dylib:+link-arg=foo

#[cfg(in_attr)]
#[link(kind = "link-arg", name = "foo")]
//[in_attr]~^ ERROR link kind `link-arg` is unstable
extern "C" {}

fn main() {}

//[in_flag]~? ERROR unknown linking modifier `link-arg`
