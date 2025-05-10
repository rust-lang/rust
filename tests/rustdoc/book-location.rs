//@ compile-flags: -Zunstable-options --book-location https://somewhere.world

#![crate_name = "foo"]

//@ has 'foo/index.html'
//@ has - '//*[@id="book-loc"]' 'Book'
//@ has - '//*[@href="https://somewhere.world"]' 'Book'
