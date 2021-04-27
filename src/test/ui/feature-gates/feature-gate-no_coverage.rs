#![crate_type = "lib"]

#[no_coverage]
#[feature(no_coverage)] // does not have to be enabled before `#[no_coverage]`
fn no_coverage_is_enabled_on_this_function() {}

#[no_coverage] //~ ERROR the `#[no_coverage]` attribute is an experimental feature
fn requires_feature_no_coverage() {}
