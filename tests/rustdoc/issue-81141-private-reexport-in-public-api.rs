// This test ensures that if a private re-export is present in a public API, it'll be
// replaced by the first public item in the re-export chain or by the private item.

#![crate_name = "foo"]

use crate::bar::Bar as Alias;

pub use crate::bar::Bar as Whatever;
use crate::Whatever as Whatever2;
use crate::Whatever2 as Whatever3;
pub use crate::bar::Inner as Whatever4;

mod bar {
    pub struct Bar;
    pub use self::Bar as Inner;
}

// @has 'foo/fn.bar.html'
// @has - '//*[@class="rust item-decl"]/code' 'pub fn bar() -> Bar'
pub fn bar() -> Alias {
    Alias
}

// @has 'foo/fn.bar2.html'
// @has - '//*[@class="rust item-decl"]/code' 'pub fn bar2() -> Whatever'
pub fn bar2() -> Whatever3 {
    Whatever
}

// @has 'foo/fn.bar3.html'
// @has - '//*[@class="rust item-decl"]/code' 'pub fn bar3() -> Whatever4'
pub fn bar3() -> Whatever4 {
    Whatever
}
