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
//@ has foo/type.Mod1Public.html
//@ !has foo/type.Mod1Private.html
//@ has foo/type.Mod2Public.html
//@ !has foo/type.Mod2Private.html

//@ has-dir foo/mod1
//@ !has foo/mod1/index.html
//@ has foo/mod1/type.Mod1Public.html
//@ !has foo/mod1/type.Mod1Private.html
//@ !has foo/mod1/type.Mod2Public.html
//@ !has foo/mod1/type.Mod2Private.html

//@ has-dir foo/mod1/mod2
//@ !has foo/mod1/mod2/index.html
//@ has foo/mod1/mod2/type.Mod2Public.html
//@ !has foo/mod1/mod2/type.Mod2Private.html

//@ !has-dir foo/mod2
//@ !has foo/mod2/index.html
//@ !has foo/mod2/type.Mod2Public.html
//@ !has foo/mod2/type.Mod2Private.html
