// rustfmt-normalize_comments: true

// Imports.

// Long import.
use exceedingly::loooooooooooooooooooooooooooooooooooooooooooooooooooooooong::import::path::{
    ItemA, ItemB,
};
use exceedingly::looooooooooooooooooooooooooooooooooooooooooooooooooooooooooong::import::path::{
    ItemA, ItemB,
};
use rustc_ast::ast::{ItemDefaultImpl, ItemForeignMod, ItemImpl, ItemMac, ItemMod, ItemStatic};

use list::{
    // Another item
    AnotherItem, // Another Comment
    // Last Item
    LastItem,
    // Some item
    SomeItem, // Comment
};

use test::{/* A */ self /* B */, Other /* C */};

pub use rustc_ast::ast::{Expr, ExprAssign, ExprCall, ExprMethodCall, ExprPath, Expr_};
use rustc_ast::{self};
use Foo::{Bar, Baz};
use {Bar /* comment */, /* Pre-comment! */ Foo};

use std::io;
use std::io::{self};

mod Foo {
    pub use rustc_ast::ast::{
        ItemDefaultImpl, ItemForeignMod, ItemImpl, ItemMac, ItemMod, ItemStatic,
    };

    mod Foo2 {
        pub use rustc_ast::ast::{
            self, ItemDefaultImpl, ItemForeignMod, ItemImpl, ItemMac, ItemMod, ItemStatic,
        };
    }
}

fn test() {
    use Baz::*;
    use Qux;
}

// Simple imports
use bar::quux as kaas;
use foo;
use foo::bar::baz;

// With aliases.
use foo::qux as bar;
use foo::{self as bar};
use foo::{self as bar, baz};
use foo::{baz, qux as bar};

// With absolute paths
use foo;
use foo::Bar;
use foo::{Bar, Baz};
use Foo;
use {Bar, Baz};

// Root globs
use *;
use *;

// spaces used to cause glob imports to disappear (#1356)
use super::*;
use foo::issue_1356::*;

// We shouldn't remove imports which have attributes attached (#1858)
#[cfg(unix)]
use self::unix::{};

// nested imports
use foo::{
    a, b,
    bar::{
        baz,
        foo::{a, b, cxxxxxxxxxxxxx, yyyyyyyyyyyyyy, zzzzzzzzzzzzzzzz},
        qux, xxxxxxxxxxx, yyyyyyyyyyyyy, zzzzzzzzzzzzzzzz,
    },
    boo, c,
};

use fooo::{
    baar::foobar::{
        xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx, yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy,
        zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz,
    },
    bar,
    bar::*,
    x, y, z,
};

use exonum::{
    api::{Api, ApiError},
    blockchain::{self, BlockProof, Blockchain, Transaction, TransactionSet},
    crypto::{Hash, PublicKey},
    helpers::Height,
    node::TransactionSend,
    storage::{ListProof, MapProof},
};

// nested imports with a single sub-tree.
use a::b::c::d;
use a::b::c::*;
use a::b::c::{xxx, yyy, zzz};

// #2645
/// This line is not affected.
// This line is deleted.
use c;

// #2670
#[macro_use]
use imports_with_attr;

// #2888
use std::f64::consts::{E, PI, SQRT_2};

// #3273
#[rustfmt::skip]
use std::fmt::{self, {Display, Formatter}};
