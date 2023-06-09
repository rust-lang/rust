// This test ensures that the `struct.B.html` only exists in `a`:
// since `a::B` is public (and inlined too), `self::a::B` doesn't
// need to be inlined as well.

#![crate_name = "foo"]

pub mod a {
    // @has 'foo/a/index.html'
    // Should only contain "Structs".
    // @count - '//*[@id="main-content"]//*[@class="item-table"]' 1
    // @has - '//*[@id="structs"]' 'Structs'
    // @has - '//*[@id="main-content"]//a[@href="struct.A.html"]' 'A'
    // @has - '//*[@id="main-content"]//a[@href="struct.B.html"]' 'B'
    mod b {
        pub struct B;
    }
    pub use self::b::B;
    pub struct A;
}

// @has 'foo/index.html'
// @!has - '//*[@id="structs"]' 'Structs'
// @has - '//*[@id="reexports"]' 'Re-exports'
// @has - '//*[@id="modules"]' 'Modules'
// @has - '//*[@id="main-content"]//*[@id="reexport.A"]' 'pub use self::a::A;'
// @has - '//*[@id="main-content"]//*[@id="reexport.B"]' 'pub use self::a::B;'
// Should only contain "Modules" and "Re-exports".
// @count - '//*[@id="main-content"]//*[@class="item-table"]' 2
pub use self::a::{A, B};
