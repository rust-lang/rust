// This test ensures that `--` (double-hyphen) is correctly converted into `–` (dash).

#![crate_name = "foo"]

//@ has 'foo/index.html' '//dd' '–'
//@ has 'foo/type.Bar.html' '//*[@class="docblock"]' '–'

/// --
pub struct Bar;
