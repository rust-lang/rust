//@ aux-crate:priv:pub_struct=pub-struct.rs
//@ compile-flags:-Z unstable-options

// The issue was an ICE which meant that we never actually generated the docs
// so if we have generated the docs, we're okay.
// Since we don't generate the docs for the auxiliary files, we can't actually
// verify that the struct is linked correctly.

//@ has issue_66159/index.html
//! [pub_struct::SomeStruct]
