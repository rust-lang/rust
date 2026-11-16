//@ compile-flags: -Zunstable-options --crate-list-heading=helloworld
#![crate_name = "foo"]

//@ hasraw crates.js '"h":"helloworld"'
//@ hasraw crates.js '"c":"foo"'
