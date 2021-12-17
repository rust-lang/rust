#![crate_name = "foo"]

// @has foo/index.html '//*[@class="item-right docblock-short"]/p' 'fooo'
// @!has foo/index.html '//*[@class="item-right docblock-short"]/p/h1' 'fooo'
// @has foo/fn.foo.html '//h2[@id="fn.foo.fooo"]/a[@href="#fn.foo.fooo"]' 'fooo'

/// # fooo
///
/// foo
pub fn foo() {}

// @has foo/index.html '//*[@class="item-right docblock-short"]/p' 'mooood'
// @!has foo/index.html '//*[@class="item-right docblock-short"]/p/h2' 'mooood'
// @has foo/foo/index.html '//h3[@id="mod.foo.mooood"]/a[@href="#mod.foo.mooood"]' 'mooood'

/// ## mooood
///
/// foo mod
pub mod foo {}

// @has foo/index.html '//*[@class="item-right docblock-short"]/p/a[@href=\
//                      "https://nougat.world"]/code' 'nougat'

/// [`nougat`](https://nougat.world)
pub struct Bar;
