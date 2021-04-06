// aux-build:issue-66159-1.rs
// aux-crate:priv:issue_66159_1=issue-66159-1.rs
// build-aux-docs
// compile-flags:-Z unstable-options

// @has extern_crate_only_used_in_link/index.html
// @has - '//a[@href="../issue_66159_1/struct.Something.html"]' 'issue_66159_1::Something'
//! [issue_66159_1::Something]
