#![crate_name = "foo"]

mod mod1 {
    mod mod2 {
        pub struct Mod2Public;
        struct Mod2Private;
    }
    pub use self::mod2::*;

    pub struct Mod1Public;
    struct Mod1Private;
}
pub use mod1::*;

//@ has foo/index.html
//@ !hasraw - "mod1"
//@ hasraw - "Mod1Public"
//@ !hasraw - "Mod1Private"
//@ !hasraw - "mod2"
//@ hasraw - "Mod2Public"
//@ !hasraw - "Mod2Private"
//@ has foo/struct.Mod1Public.html
//@ !has foo/struct.Mod1Private.html
//@ has foo/struct.Mod2Public.html
//@ !has foo/struct.Mod2Private.html

//@ has-dir foo/mod1
//@ !has foo/mod1/index.html
//@ has foo/mod1/struct.Mod1Public.html
//@ !has foo/mod1/struct.Mod1Private.html
//@ !has foo/mod1/struct.Mod2Public.html
//@ !has foo/mod1/struct.Mod2Private.html

//@ has-dir foo/mod1/mod2
//@ !has foo/mod1/mod2/index.html
//@ has foo/mod1/mod2/struct.Mod2Public.html
//@ !has foo/mod1/mod2/struct.Mod2Private.html

//@ !has-dir foo/mod2
//@ !has foo/mod2/index.html
//@ !has foo/mod2/struct.Mod2Public.html
//@ !has foo/mod2/struct.Mod2Private.html
