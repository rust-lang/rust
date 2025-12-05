// Issue #62863
// ignore-tidy-cr

// Note: if you see ^M in this file, that's how your editor renders literal `\r`

/// This doc comment contains three isolated `\r` symbols
//~^ ERROR bare CR not allowed in doc-comment
//~| ERROR bare CR not allowed in doc-comment
//~| ERROR bare CR not allowed in doc-comment
fn main() {}
