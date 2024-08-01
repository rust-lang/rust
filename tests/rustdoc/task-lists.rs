// ignore-tidy-linelength
// FIXME: this doesn't test as much as I'd like; ideally it would have these query too:
  // has task_lists/index.html '//li/input[@type="checkbox" and disabled]/following-sibling::text()' 'a'
  // has task_lists/index.html '//li/input[@type="checkbox"]/following-sibling::text()' 'b'
// Unfortunately that requires LXML, because the built-in xml module doesn't support all of xpath.

//@ has task_lists/index.html '//ul/li/input[@type="checkbox"]' ''
//@ has task_lists/index.html '//ul/li/input[@disabled]' ''
//@ has task_lists/index.html '//ul/li' 'a'
//@ has task_lists/index.html '//ul/li' 'b'
//! This tests 'task list' support, a common markdown extension.
//! - [ ] a
//! - [x] b
