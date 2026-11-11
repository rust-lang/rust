// This test ensures that `--` (double-hyphen) is correctly converted into `–` (dash).

#![crate_name = "foo"]

//@ has 'foo/index.html' '//dd' '–'
//@ has 'foo/struct.Bar.html' '//*[@class="docblock"]' '–'

/// --
pub struct Bar;
