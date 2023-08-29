//// borked doc comment on the first line. doesn't combust!
fn a() {}

// This test's entire purpose is to make sure we don't panic if the comment with four slashes
// extends to the first line of the file. This is likely pretty rare in production, but an ICE is an
// ICE.
