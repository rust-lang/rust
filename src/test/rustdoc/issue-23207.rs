// aux-build:issue-23207-1.rs
// aux-build:issue-23207-2.rs
// ignore-cross-compile

extern crate issue_23207_2;

// @has issue_23207/fmt/index.html
// @count - '//*[@class="struct"]' 1
pub use issue_23207_2::fmt;
