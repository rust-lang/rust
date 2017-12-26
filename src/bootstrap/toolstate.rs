// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[derive(Copy, Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
/// Whether a tool can be compiled, tested or neither
pub enum ToolState {
    /// The tool compiles successfully, but the test suite fails
    TestFail = 1,
    /// The tool compiles successfully and its test suite passes
    TestPass = 2,
    /// The tool can't even be compiled
    BuildFail = 0,
}

impl Default for ToolState {
    fn default() -> Self {
        // err on the safe side
        ToolState::BuildFail
    }
}
