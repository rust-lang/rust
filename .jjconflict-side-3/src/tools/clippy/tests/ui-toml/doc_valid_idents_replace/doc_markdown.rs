#![warn(clippy::doc_markdown)]

/// This is a special interface for ClipPy which doesn't require backticks
fn allowed_name() {}

/// OAuth and LaTeX are inside Clippy's default list.
//~^ doc_markdown
//~| doc_markdown
fn default_name() {}

/// TestItemThingyOfCoolness might sound cool but is not on the list and should be linted.
//~^ doc_markdown
fn unknown_name() {}

fn main() {}
