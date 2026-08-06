//@ compile-flags: '--crate-version=1.3.37-nightly (203c57dbe 2023-09-17)'

#![crate_name="foo"]

// main version next to logo, extra version data below it
//@ has 'foo/index.html' '//h2/span[@class="version"]' '1.3.37-nightly'
//@ has 'foo/index.html' '//nav[@class="sidebar"]/div[@class="version"]' '(203c57dbe 2023-09-17)'
