// Imports.

// Long import.
use syntax::ast::{ItemForeignMod, ItemImpl, ItemMac, ItemMod, ItemStatic, ItemDefaultImpl};
use exceedingly::looooooooooooooooooooooooooooooooooooooooooooooooooooooooooong::import::path::{ItemA,
                                                                                                ItemB};

use {Foo, Bar};
use Foo::{Bar, Baz};
pub use syntax::ast::{Expr_, Expr, ExprAssign, ExprCall, ExprMethodCall, ExprPath};

mod Foo {
    pub use syntax::ast::{ItemForeignMod, ItemImpl, ItemMac, ItemMod, ItemStatic, ItemDefaultImpl};

    mod Foo2 {
        pub use syntax::ast::{ItemForeignMod, ItemImpl, ItemMac, ItemMod,
                              ItemStatic, ItemDefaultImpl};
    }
}

fn test() {
    use Baz::*;
}
