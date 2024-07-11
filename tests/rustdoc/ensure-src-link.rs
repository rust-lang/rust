#![crate_name = "foo"]

// This test ensures that the [src] link is present on traits items.

//@ has foo/trait.Iterator.html '//*[@id="method.zip"]//a[@class="src"]' "source"
pub use std::iter::Iterator;
