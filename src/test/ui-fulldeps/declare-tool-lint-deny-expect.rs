// compile-flags: -Z unstable-options

#![feature(rustc_private)]

extern crate rustc_session;

use rustc_session::declare_tool_lint;

declare_tool_lint! {
    pub clippy::TEST_LINT_ALLOW,
    Allow,
    "test"
}

// This should cause an error
declare_tool_lint! {
    pub clippy::TEST_LINT_EXPECT,
    Expect,
    "test"
}
//~^^^^^ ERROR: `Expect` is not allowed as an initial level for lints

declare_tool_lint! {
    pub clippy::TEST_LINT_WARN,
    Warn,
    "test"
}

declare_tool_lint! {
    pub clippy::TEST_LINT_DENY,
    Deny,
    "test"
}

fn main() {}
