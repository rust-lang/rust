// This test is just a little cursed.
//@ aux-build:issue-66159-1.rs
//@ aux-crate:priv:issue_66159_1=issue-66159-1.rs
//@ aux-build:empty.rs
//@ aux-crate:priv:empty=empty.rs
//@ aux-build:empty2.rs
//@ aux-crate:priv:empty2=empty2.rs
//@ build-aux-docs
//@ compile-flags:-Z unstable-options
//@ edition: 2018

//@ has extern_crate_only_used_in_link/index.html
//@ has - '//a[@href="../issue_66159_1/struct.Something.html"]' 'issue_66159_1::Something'
//! [issue_66159_1::Something]

//@ has - '//a[@href="../empty/index.html"]' 'empty'
//! [`empty`]

//@ has - '//a[@href="../empty2/index.html"]' 'empty2'
//! [`empty2<x>`]
