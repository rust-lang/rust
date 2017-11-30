// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use build_helper::BuildExpectation;

#[derive(Copy, Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
/// Whether a tool can be compiled, tested or neither
pub enum ToolState {
    /// The tool compiles successfully, but the test suite fails
    Compiling = 1,
    /// The tool compiles successfully and its test suite passes
    Testing = 2,
    /// The tool can't even be compiled
    Broken = 0,
}

impl ToolState {
    /// If a tool with the current toolstate should be working on
    /// the given toolstate
    pub fn passes(self, other: ToolState) -> BuildExpectation {
        if self as usize >= other as usize {
            BuildExpectation::Succeeding
        } else {
            BuildExpectation::Failing
        }
    }

    pub fn testing(&self) -> bool {
        match *self {
            ToolState::Testing => true,
            _ => false,
        }
    }
}

impl Default for ToolState {
    fn default() -> Self {
        // err on the safe side
        ToolState::Broken
    }
}

#[derive(Copy, Clone, Debug, Deserialize, Default)]
/// Used to express which tools should (not) be compiled or tested.
/// This is created from `toolstate.toml`.
pub struct ToolStates {
    pub miri: ToolState,
    pub clippy: ToolState,
    pub rls: ToolState,
    pub rustfmt: ToolState,
}
