//@ ignore-auxiliary (used by `./circular-module-with-doc-comment-issue-97589.rs`)

//! this comment caused the circular dependency checker to break

#[path = "recursive.rs"]
mod recursive;
