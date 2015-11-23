// Imports.

// Long import.
use syntax::ast::{ItemForeignMod, ItemImpl, ItemMac, ItemMod, ItemStatic, ItemDefaultImpl};
use exceedingly::looooooooooooooooooooooooooooooooooooooooooooooooooooooooooong::import::path::{ItemA,
                                                                                                ItemB};
use exceedingly::loooooooooooooooooooooooooooooooooooooooooooooooooooooooong::import::path::{ItemA,
                                                                                             ItemB};

use list::{// Some item
           SomeItem, // Comment
           // Another item
           AnotherItem, // Another Comment
           // Last Item
           LastItem};

use test::{/* A */ self /* B */, Other /* C */};

use syntax;
use {/* Pre-comment! */ Foo, Bar /* comment */};
use Foo::{Bar, Baz};
pub use syntax::ast::{Expr_, Expr, ExprAssign, ExprCall, ExprMethodCall, ExprPath};

mod Foo {
    pub use syntax::ast::{ItemForeignMod, ItemImpl, ItemMac, ItemMod, ItemStatic, ItemDefaultImpl};

    mod Foo2 {
        pub use syntax::ast::{self, ItemForeignMod, ItemImpl, ItemMac, ItemMod, ItemStatic,
                              ItemDefaultImpl};
    }
}

fn test() {
    use Baz::*;
    use Qux;
}

// Simple imports
use foo::bar::baz;
use bar::quux as kaas;
use foo;

// With aliases.
use foo::{self as bar, baz};
use foo as bar;
use foo::qux as bar;
use foo::{baz, qux as bar};
