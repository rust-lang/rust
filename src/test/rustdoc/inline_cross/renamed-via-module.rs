// aux-build:renamed-via-module.rs
// build-aux-docs
// ignore-cross-compile

#![crate_name = "bar"]

extern crate foo;

// @has foo/iter/index.html
// @has - '//a/[@href="struct.DeprecatedStepBy.html"]' "DeprecatedStepBy"
// @has - '//a/[@href="struct.StepBy.html"]' "StepBy"
// @has foo/iter/struct.DeprecatedStepBy.html
// @has - '//h1' "Struct foo::iter::DeprecatedStepBy"
// @has foo/iter/struct.StepBy.html
// @has - '//h1' "Struct foo::iter::StepBy"

// @has bar/iter/index.html
// @has - '//a/[@href="struct.DeprecatedStepBy.html"]' "DeprecatedStepBy"
// @has - '//a/[@href="struct.StepBy.html"]' "StepBy"
// @has bar/iter/struct.DeprecatedStepBy.html
// @has - '//h1' "Struct bar::iter::DeprecatedStepBy"
// @has bar/iter/struct.StepBy.html
// @has - '//h1' "Struct bar::iter::StepBy"
pub use foo::iter;
