// This test ensures that all re-exports of doc hidden elements are displayed.

#![crate_name = "foo"]

#[doc(hidden)]
pub struct Bar;

#[macro_export]
#[doc(hidden)]
macro_rules! foo {
    () => {};
}

// @has 'foo/index.html'
// @has - '//*[@id="reexport.Macro"]/code' 'pub use crate::foo as Macro;'
pub use crate::foo as Macro;
// @has - '//*[@id="reexport.Macro2"]/code' 'pub use crate::foo as Macro2;'
pub use crate::foo as Macro2;
// @has - '//*[@id="reexport.Boo"]/code' 'pub use crate::Bar as Boo;'
pub use crate::Bar as Boo;
// @has - '//*[@id="reexport.Boo2"]/code' 'pub use crate::Bar as Boo2;'
pub use crate::Bar as Boo2;

pub fn fofo() {}

// @has - '//*[@id="reexport.f1"]/code' 'pub use crate::fofo as f1;'
pub use crate::fofo as f1;
// @has - '//*[@id="reexport.f2"]/code' 'pub use crate::fofo as f2;'
pub use crate::fofo as f2;

pub mod sub {
    // @has 'foo/sub/index.html'
    // @has - '//*[@id="reexport.Macro"]/code' 'pub use crate::foo as Macro;'
    pub use crate::foo as Macro;
    // @has - '//*[@id="reexport.Macro2"]/code' 'pub use crate::foo as Macro2;'
    pub use crate::foo as Macro2;

    // @has - '//*[@id="reexport.f1"]/code' 'pub use crate::fofo as f1;'
    pub use crate::fofo as f1;
    // @has - '//*[@id="reexport.f2"]/code' 'pub use crate::fofo as f2;'
    pub use crate::fofo as f2;
}
