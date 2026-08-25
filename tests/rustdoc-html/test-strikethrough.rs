#![crate_name = "foo"]

// Test that strikethrough works with single and double tildes and that it shows up on
// the item's dedicated page as well as the parent module's summary of items.

//@ has foo/index.html //del 'strike'
//@ has foo/index.html //del 'through'

//@ has foo/fn.f.html //del 'strike'
//@ has foo/fn.f.html //del 'through'

/// ~~strike~~ ~through~
pub fn f() {}
