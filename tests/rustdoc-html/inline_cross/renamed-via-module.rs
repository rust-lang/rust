//@ aux-build:renamed-via-module.rs
//@ build-aux-docs
//@ ignore-cross-compile

#![crate_name = "bar"]

extern crate foo;

//@ has foo/iter/index.html
//@ has - '//a/[@href="type.DeprecatedStepBy.html"]' "DeprecatedStepBy"
//@ has - '//a/[@href="type.StepBy.html"]' "StepBy"
//@ has foo/iter/type.DeprecatedStepBy.html
//@ has - '//h1' "Struct DeprecatedStepBy"
//@ matches - '//*[@class="rustdoc-breadcrumbs"]' 'foo::iter'
//@ has foo/iter/type.StepBy.html
//@ has - '//h1' "Struct StepBy"
//@ matches - '//*[@class="rustdoc-breadcrumbs"]' 'foo::iter'

//@ has bar/iter/index.html
//@ has - '//a/[@href="type.DeprecatedStepBy.html"]' "DeprecatedStepBy"
//@ has - '//a/[@href="type.StepBy.html"]' "StepBy"
//@ has bar/iter/type.DeprecatedStepBy.html
//@ has - '//h1' "Struct DeprecatedStepBy"
//@ matches - '//*[@class="rustdoc-breadcrumbs"]' 'bar::iter'
//@ has bar/iter/type.StepBy.html
//@ has - '//h1' "Struct StepBy"
//@ matches - '//*[@class="rustdoc-breadcrumbs"]' 'bar::iter'
pub use foo::iter;
