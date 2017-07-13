// rustfmt-normalize_comments: true
// rustfmt-error_on_line_overflow: false

// Imports.

// Long import.
use syntax::ast::{ItemDefaultImpl, ItemForeignMod, ItemImpl, ItemMac, ItemMod, ItemStatic};
use exceedingly::looooooooooooooooooooooooooooooooooooooooooooooooooooooooooong::import::path::{ItemA,
                                                                                                ItemB};
use exceedingly::loooooooooooooooooooooooooooooooooooooooooooooooooooooooong::import::path::{ItemA,
                                                                                             ItemB};

use list::{// Another item
           AnotherItem, // Another Comment
           // Last Item
           LastItem,
           // Some item
           SomeItem /* Comment */};

use test::{/* A */ self /* B */, Other /* C */};

use syntax;
use {Bar /* comment */, /* Pre-comment! */ Foo};
use Foo::{Bar, Baz};
pub use syntax::ast::{Expr, ExprAssign, ExprCall, ExprMethodCall, ExprPath, Expr_};

use self;
use std::io;
use std::io;

mod Foo {
    pub use syntax::ast::{ItemDefaultImpl, ItemForeignMod, ItemImpl, ItemMac, ItemMod, ItemStatic};

    mod Foo2 {
        pub use syntax::ast::{self, ItemDefaultImpl, ItemForeignMod, ItemImpl, ItemMac, ItemMod,
                              ItemStatic};
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

// With absolute paths
use foo;
use foo::Bar;
use foo::{Bar, Baz};
use Foo;
use {Bar, Baz};

// Root globs
use ::*;
use ::*;

// spaces used to cause glob imports to disappear (#1356)
use super::*;
use foo::issue_1356::*;
