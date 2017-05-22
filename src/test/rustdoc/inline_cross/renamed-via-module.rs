// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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
