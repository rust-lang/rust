// edition:2015

#![crate_name = "foo"]

use external::Public as Private;

pub mod external {
    pub struct Public;

    // @has 'foo/external/fn.make.html'
    // @has - '//*[@class="rust item-decl"]/code' 'pub fn make() -> Public'
    pub fn make() -> ::Private { super::Private }
}
